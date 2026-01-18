"""
Passkey retrieval evaluation for testing contextual knowledge understanding.

The passkey task embeds a random code in a long context and tests whether
the model can retrieve it - a pure test of in-context memory.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


@dataclass
class PasskeyConfig:
    """Configuration for passkey retrieval task."""

    # Context parameters
    context_length: int = 1024
    num_passkeys: int = 1

    # Passkey parameters
    passkey_length: int = 5
    passkey_chars: str = "0123456789"

    # Position parameters (where to insert passkey)
    depth_percent: float = 50.0  # Percentage through context

    # Evaluation parameters
    num_samples: int = 100

    # Filler text
    filler_text: str = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


class PasskeyRetrievalEvaluator:
    """
    Evaluator for the passkey retrieval task.

    This task tests whether models can retrieve specific information
    embedded in a long context, which requires contextual knowledge
    understanding (not parametric knowledge).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: PasskeyConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def _generate_passkey(self) -> str:
        """Generate a random passkey."""
        return "".join(
            random.choice(self.config.passkey_chars)
            for _ in range(self.config.passkey_length)
        )

    def _create_context(
        self,
        passkey: str,
        depth_percent: float,
    ) -> Tuple[str, str]:
        """
        Create a context with embedded passkey.

        Returns:
            Tuple of (full_context, prompt_for_retrieval)
        """
        # Calculate target token position
        target_tokens = int(self.config.context_length * depth_percent / 100)

        # Build filler to reach target position
        filler_tokens = self.tokenizer.encode(self.config.filler_text, add_special_tokens=False)
        tokens_per_filler = len(filler_tokens)
        num_fillers_before = target_tokens // tokens_per_filler

        filler_before = self.config.filler_text * num_fillers_before

        # Passkey insertion text
        passkey_text = f"The pass key is {passkey}. Remember it. {passkey} is the pass key. "

        # Fill remaining context
        remaining_tokens = self.config.context_length - target_tokens - len(
            self.tokenizer.encode(passkey_text, add_special_tokens=False)
        )
        num_fillers_after = remaining_tokens // tokens_per_filler

        filler_after = self.config.filler_text * num_fillers_after

        # Combine
        context = filler_before + passkey_text + filler_after

        # Create retrieval prompt
        prompt = context + "\nWhat is the pass key? The pass key is "

        return context, prompt

    def _extract_prediction(self, generated_text: str) -> str:
        """Extract the predicted passkey from generated text."""
        # Look for digits at the start of the generated continuation
        prediction = ""
        for char in generated_text:
            if char in self.config.passkey_chars:
                prediction += char
                if len(prediction) >= self.config.passkey_length:
                    break
            elif prediction:
                # Stop at first non-digit after starting to collect
                break
        return prediction

    def evaluate_single(
        self,
        depth_percent: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Evaluate a single passkey retrieval.

        Args:
            depth_percent: Where to place passkey (0-100). Uses config default if None.

        Returns:
            Dict with passkey, prediction, correct, depth
        """
        if depth_percent is None:
            depth_percent = self.config.depth_percent

        passkey = self._generate_passkey()
        context, prompt = self._create_context(passkey, depth_percent)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_length + 50,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.passkey_length + 5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        prediction = self._extract_prediction(generated)
        correct = prediction == passkey

        return {
            "passkey": passkey,
            "prediction": prediction,
            "correct": correct,
            "depth_percent": depth_percent,
            "context_length": self.config.context_length,
        }

    def evaluate(
        self,
        depths: Optional[List[float]] = None,
        num_samples_per_depth: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Run full evaluation across multiple depths.

        Args:
            depths: List of depth percentages to test. Default: [10, 25, 50, 75, 90]
            num_samples_per_depth: Samples per depth. Uses config.num_samples if None.

        Returns:
            Dict with overall accuracy and per-depth breakdown
        """
        if depths is None:
            depths = [10, 25, 50, 75, 90]

        if num_samples_per_depth is None:
            num_samples_per_depth = self.config.num_samples // len(depths)

        self.model.eval()
        results = []

        for depth in tqdm(depths, desc="Depth"):
            depth_results = []
            for _ in tqdm(range(num_samples_per_depth), desc=f"Depth {depth}%", leave=False):
                result = self.evaluate_single(depth_percent=depth)
                depth_results.append(result)

            results.extend(depth_results)

        # Compute statistics
        overall_correct = sum(r["correct"] for r in results)
        overall_accuracy = overall_correct / len(results)

        per_depth = {}
        for depth in depths:
            depth_results = [r for r in results if r["depth_percent"] == depth]
            depth_correct = sum(r["correct"] for r in depth_results)
            per_depth[depth] = {
                "accuracy": depth_correct / len(depth_results),
                "num_samples": len(depth_results),
            }

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": len(results),
            "context_length": self.config.context_length,
            "per_depth": per_depth,
            "all_results": results,
        }


def run_passkey_sweep(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context_lengths: List[int] = [512, 1024, 2048, 4096],
    depths: List[float] = [10, 50, 90],
    num_samples_per_config: int = 20,
) -> Dict[str, Dict]:
    """
    Run passkey retrieval across multiple context lengths.

    Returns nested dict: {context_length: {depth: accuracy}}
    """
    results = {}

    for ctx_len in tqdm(context_lengths, desc="Context lengths"):
        config = PasskeyConfig(
            context_length=ctx_len,
            num_samples=num_samples_per_config * len(depths),
        )
        evaluator = PasskeyRetrievalEvaluator(model, tokenizer, config)
        eval_results = evaluator.evaluate(depths=depths, num_samples_per_depth=num_samples_per_config)

        results[ctx_len] = {
            "overall_accuracy": eval_results["overall_accuracy"],
            "per_depth": eval_results["per_depth"],
        }

    return results
