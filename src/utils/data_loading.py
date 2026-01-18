"""
Data loading utilities for training and evaluation.
"""

from typing import Dict, Iterator, List, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset


class PretrainingDataset(IterableDataset):
    """
    Streaming dataset for pretraining/recalibration.

    Uses the same data distribution as original pretraining,
    following the DroPE paper's approach.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "HuggingFaceTB/smollm-corpus",
        dataset_config: Optional[str] = "fineweb-edu-dedup",
        split: str = "train",
        max_length: int = 2048,
        streaming: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        buffer_len = 0

        for example in self.dataset:
            # Get text from example
            if "text" in example:
                text = example["text"]
            elif "content" in example:
                text = example["content"]
            else:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer_len = len(buffer)

            # Yield chunks of max_length
            while buffer_len >= self.max_length:
                chunk = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                buffer_len = len(buffer)

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(len(chunk), dtype=torch.long),
                }


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation tasks like IMDB sentiment classification.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "test",
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        example = self.dataset[idx]
        text = example[self.text_column]
        label = example[self.label_column]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "text": text,
        }


def create_pretraining_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 4,
    dataset_name: str = "HuggingFaceTB/smollm-corpus",
    dataset_config: str = "fineweb-edu-dedup",
) -> DataLoader:
    """
    Create a DataLoader for pretraining/recalibration.
    """
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_evaluation_dataloader(
    tokenizer: PreTrainedTokenizer,
    task: str,
    batch_size: int = 8,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for evaluation tasks.

    Args:
        tokenizer: The tokenizer
        task: Task name ("imdb", "gsm8k", etc.)
        batch_size: Batch size
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to load

    Returns:
        DataLoader for the task
    """
    task_configs = {
        "imdb": {
            "dataset_name": "imdb",
            "split": "test",
            "text_column": "text",
            "label_column": "label",
        },
        "gsm8k": {
            "dataset_name": "gsm8k",
            "dataset_config": "main",
            "split": "test",
            "text_column": "question",
            "label_column": "answer",
        },
        "aqua": {
            "dataset_name": "aqua_rat",
            "split": "test",
            "text_column": "question",
            "label_column": "correct",
        },
    }

    if task not in task_configs:
        raise ValueError(f"Unknown task: {task}. Available: {list(task_configs.keys())}")

    config = task_configs[task]

    dataset = EvaluationDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        **config,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
