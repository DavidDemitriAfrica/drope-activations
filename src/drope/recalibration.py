"""
Recalibration training for DroPE models.

After removing RoPE, the model needs brief continued pretraining to adapt
to the absence of positional information.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import wandb


@dataclass
class RecalibrationConfig:
    """Configuration for DroPE recalibration training."""

    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Data parameters
    max_seq_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Training duration
    num_tokens: int = 10_000_000_000  # 10B tokens default
    save_every_tokens: int = 1_000_000_000  # Save every 1B tokens

    # Checkpointing
    output_dir: str = "./drope_checkpoints"
    save_total_limit: int = 5

    # Logging
    logging_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = "drope-recalibration"

    # Optional QKNorm for stability
    use_qknorm: bool = False


class DroPERecalibrationTrainer:
    """
    Trainer for recalibrating DroPE models.

    Following the DroPE paper, this performs continued pretraining on the
    original training data distribution after removing RoPE.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RecalibrationConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.device = next(model.parameters()).device
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Compute total steps
        tokens_per_step = config.batch_size * config.max_seq_length * config.gradient_accumulation_steps
        self.total_steps = config.num_tokens // tokens_per_step

        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * config.warmup_ratio),
            num_training_steps=self.total_steps,
        )

        # Tracking
        self.global_step = 0
        self.tokens_seen = 0
        self.checkpoint_callbacks: List[Callable] = []

    def add_checkpoint_callback(self, callback: Callable):
        """Add callback to be called at each checkpoint save."""
        self.checkpoint_callbacks.append(callback)

    def train(self):
        """Run recalibration training."""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
            )

        self.model.train()
        progress_bar = tqdm(total=self.total_steps, desc="Recalibration")

        accumulation_loss = 0
        accumulation_steps = 0

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.total_steps:
            # Get batch (cycle through dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # Causal LM objective
            )
            loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            accumulation_loss += loss.item()
            accumulation_steps += 1

            # Update tokens seen
            self.tokens_seen += input_ids.numel()

            # Gradient accumulation step
            if accumulation_steps >= self.config.gradient_accumulation_steps:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                progress_bar.update(1)

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = accumulation_loss / accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]

                    log_dict = {
                        "loss": avg_loss,
                        "learning_rate": lr,
                        "tokens_seen": self.tokens_seen,
                        "step": self.global_step,
                    }

                    if self.config.use_wandb:
                        wandb.log(log_dict)

                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        tokens=f"{self.tokens_seen/1e9:.2f}B",
                    )

                accumulation_loss = 0
                accumulation_steps = 0

                # Checkpointing
                save_every_steps = self.config.save_every_tokens // (
                    self.config.batch_size * self.config.max_seq_length * self.config.gradient_accumulation_steps
                )
                if self.global_step % save_every_steps == 0:
                    self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint(final=True)
        progress_bar.close()

        if self.config.use_wandb:
            wandb.finish()

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_name = "final"
        else:
            checkpoint_name = f"checkpoint-{self.tokens_seen//1_000_000_000}B"

        checkpoint_path = self.output_dir / checkpoint_name

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, checkpoint_path / "training_state.pt")

        # Call callbacks (e.g., for massive value analysis)
        for callback in self.checkpoint_callbacks:
            callback(self.model, checkpoint_path, self.tokens_seen)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1].rstrip("B")),
        )

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataloader."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                total_loss += outputs.loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()

        self.model.train()

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
