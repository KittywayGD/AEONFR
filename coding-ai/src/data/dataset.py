"""Dataset management for code LLM training.

This module handles both initial training data and dynamically generated
samples for recursive learning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset for code samples.

    This dataset handles tokenization and batching of code samples.

    Attributes:
        samples: List of code samples.
        tokenizer: Tokenizer for encoding text.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        samples: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        """Initialize code dataset.

        Args:
            samples: List of code strings.
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(
            f"Initialized CodeDataset with {len(samples)} samples, "
            f"max_length={max_length}"
        )

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item.

        Returns:
            Dictionary containing input_ids, attention_mask, and labels.
        """
        code = self.samples[idx]

        # Encode the code
        token_ids = self.tokenizer.encode(code, add_special_tokens=True)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            # Pad with pad token
            pad_token_id = self.tokenizer.pad_token_id or 0
            token_ids = token_ids + [pad_token_id] * (self.max_length - len(token_ids))

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != (self.tokenizer.pad_token_id or 0) else 0
                          for token_id in token_ids]

        # For language modeling, labels are the same as input_ids
        labels = token_ids.copy()

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class StreamingCodeDataset(IterableDataset):
    """Streaming dataset for large code corpora.

    This dataset streams data from HuggingFace datasets without loading
    everything into memory at once.

    Attributes:
        dataset_name: Name of the HuggingFace dataset.
        dataset_config: Configuration/subset of the dataset.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        split: Dataset split to use.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        streaming: bool = True,
    ):
        """Initialize streaming code dataset.

        Args:
            dataset_name: HuggingFace dataset name.
            dataset_config: Dataset configuration/subset.
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
            split: Dataset split to use.
            streaming: Whether to stream the dataset.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.streaming = streaming

        logger.info(
            f"Initializing StreamingCodeDataset: {dataset_name}/{dataset_config}"
        )

        # Load dataset
        try:
            self.dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=streaming,
            )
            logger.info(f"Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset.

        Yields:
            Dictionary containing input_ids, attention_mask, and labels.
        """
        for sample in self.dataset:
            # Extract code from sample
            # Adjust this based on the dataset structure
            if "content" in sample:
                code = sample["content"]
            elif "text" in sample:
                code = sample["text"]
            elif "code" in sample:
                code = sample["code"]
            else:
                logger.warning(f"Unknown sample structure: {sample.keys()}")
                continue

            # Skip empty or too short samples
            if not code or len(code.strip()) < 10:
                continue

            # Encode
            token_ids = self.tokenizer.encode(code, add_special_tokens=True)

            # Truncate or pad
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                pad_token_id = self.tokenizer.pad_token_id or 0
                token_ids = token_ids + [pad_token_id] * (self.max_length - len(token_ids))

            # Create attention mask
            attention_mask = [1 if token_id != (self.tokenizer.pad_token_id or 0) else 0
                              for token_id in token_ids]

            # Labels
            labels = token_ids.copy()

            yield {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


class DynamicDatasetManager:
    """Manages dynamically generated samples for recursive learning.

    This class handles storing, loading, and versioning of code samples
    generated by the model during recursive self-improvement.

    Attributes:
        dataset_path: Path to the dynamic dataset file.
        max_samples: Maximum number of samples to keep.
        version: Current dataset version.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        max_samples: int = 100000,
        version: int = 1,
    ):
        """Initialize dynamic dataset manager.

        Args:
            dataset_path: Path to dataset file (JSONL format).
            max_samples: Maximum number of samples to store.
            version: Initial dataset version.
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_samples = max_samples
        self.version = version
        self.samples: List[Dict] = []

        # Load existing samples if file exists
        if self.dataset_path.exists():
            self.load()

        logger.info(
            f"DynamicDatasetManager initialized: {len(self.samples)} samples, "
            f"version={self.version}"
        )

    def add_sample(
        self,
        code: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a new sample to the dynamic dataset.

        Args:
            code: The code sample to add.
            metadata: Optional metadata about the sample.
        """
        sample = {
            "code": code,
            "version": self.version,
            "metadata": metadata or {},
        }

        self.samples.append(sample)

        # Trim if exceeding max samples
        if len(self.samples) > self.max_samples:
            # Remove oldest samples
            self.samples = self.samples[-self.max_samples:]
            logger.info(f"Trimmed dataset to {self.max_samples} most recent samples")

    def add_samples_batch(
        self,
        samples: List[Dict[str, any]],
    ) -> None:
        """Add multiple samples at once.

        Args:
            samples: List of sample dictionaries with 'code' and optional 'metadata'.
        """
        for sample_data in samples:
            code = sample_data.get("code")
            metadata = sample_data.get("metadata", {})
            if code:
                self.add_sample(code, metadata)

        logger.info(f"Added {len(samples)} samples to dynamic dataset")

    def save(self) -> None:
        """Save the dynamic dataset to disk."""
        try:
            with open(self.dataset_path, "w") as f:
                for sample in self.samples:
                    f.write(json.dumps(sample) + "\n")

            logger.info(
                f"Saved {len(self.samples)} samples to {self.dataset_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save dynamic dataset: {e}")
            raise

    def load(self) -> None:
        """Load the dynamic dataset from disk."""
        try:
            self.samples = []
            with open(self.dataset_path, "r") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        self.samples.append(sample)

            # Update version to the max version in loaded samples
            if self.samples:
                self.version = max(s.get("version", 1) for s in self.samples)

            logger.info(
                f"Loaded {len(self.samples)} samples from {self.dataset_path}, "
                f"version={self.version}"
            )
        except Exception as e:
            logger.error(f"Failed to load dynamic dataset: {e}")
            raise

    def get_samples(
        self,
        version: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Get code samples from the dataset.

        Args:
            version: Filter by specific version (None for all versions).
            limit: Maximum number of samples to return.

        Returns:
            List of code strings.
        """
        samples = self.samples

        # Filter by version
        if version is not None:
            samples = [s for s in samples if s.get("version") == version]

        # Apply limit
        if limit is not None:
            samples = samples[:limit]

        return [s["code"] for s in samples]

    def get_dataset(
        self,
        tokenizer,
        max_length: int = 512,
        version: Optional[int] = None,
    ) -> CodeDataset:
        """Create a CodeDataset from dynamic samples.

        Args:
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
            version: Filter by specific version.

        Returns:
            CodeDataset instance.
        """
        samples = self.get_samples(version=version)
        return CodeDataset(samples, tokenizer, max_length)

    def increment_version(self) -> None:
        """Increment the dataset version."""
        self.version += 1
        logger.info(f"Dataset version incremented to {self.version}")

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the dynamic dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            "total_samples": len(self.samples),
            "current_version": self.version,
            "versions": {},
        }

        # Count samples per version
        for sample in self.samples:
            version = sample.get("version", 1)
            stats["versions"][version] = stats["versions"].get(version, 0) + 1

        return stats


def create_dataloader(
    dataset: Union[CodeDataset, StreamingCodeDataset],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for code datasets.

    Args:
        dataset: Dataset to create loader from.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle (only for non-streaming datasets).

    Returns:
        DataLoader instance.
    """
    # Streaming datasets don't support shuffle
    if isinstance(dataset, IterableDataset):
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
