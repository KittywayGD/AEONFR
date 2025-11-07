"""Custom BPE tokenizer optimized for code.

This module implements a Byte-Pair Encoding (BPE) tokenizer specifically
designed for programming languages, with special handling for code-specific
patterns like indentation, identifiers, and syntax.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)


class CodeTokenizer:
    """BPE tokenizer optimized for code.

    This tokenizer is designed to handle code-specific patterns and preserve
    important structural elements like indentation and syntax.

    Attributes:
        tokenizer: The underlying tokenizers.Tokenizer object.
        vocab_size: Size of the vocabulary.
        special_tokens: Dictionary of special tokens.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2,
    ):
        """Initialize the CodeTokenizer.

        Args:
            vocab_size: Target vocabulary size.
            special_tokens: List of special tokens to add.
            min_frequency: Minimum frequency for a token to be included.
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Default special tokens
        if special_tokens is None:
            special_tokens = [
                "<|endoftext|>",
                "<|pad|>",
                "<|bos|>",
                "<|eos|>",
                "<|code|>",
                "<|comment|>",
                "<|docstring|>",
            ]

        self.special_tokens = {
            token: idx for idx, token in enumerate(special_tokens)
        }

        # Initialize tokenizer
        self.tokenizer = self._create_tokenizer()
        logger.info(
            f"Initialized CodeTokenizer with vocab_size={vocab_size}, "
            f"special_tokens={len(special_tokens)}"
        )

    def _create_tokenizer(self) -> Tokenizer:
        """Create and configure the BPE tokenizer.

        Returns:
            Configured Tokenizer object.
        """
        # Initialize BPE model
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

        # Normalizer: minimal normalization for code
        # We use NFKC to normalize unicode but preserve most structure
        tokenizer.normalizer = Sequence([NFKC()])

        # Pre-tokenizer: split on whitespace and punctuation but preserve code structure
        # ByteLevel handles all bytes and preserves whitespace information
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor: add special tokens
        special_tokens_list = list(self.special_tokens.keys())
        if "<|bos|>" in self.special_tokens and "<|eos|>" in self.special_tokens:
            tokenizer.post_processor = TemplateProcessing(
                single="<|bos|> $A <|eos|>",
                special_tokens=[
                    ("<|bos|>", self.special_tokens["<|bos|>"]),
                    ("<|eos|>", self.special_tokens["<|eos|>"]),
                ],
            )

        return tokenizer

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: Optional[int] = None,
    ) -> None:
        """Train the tokenizer on a corpus of code.

        Args:
            files: Path(s) to training files.
            vocab_size: Override the default vocabulary size.
        """
        if isinstance(files, str):
            files = [files]

        vocab_size = vocab_size or self.vocab_size

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=list(self.special_tokens.keys()),
            show_progress=True,
        )

        logger.info(f"Training tokenizer on {len(files)} files...")
        self.tokenizer.train(files=files, trainer=trainer)
        logger.info(f"Tokenizer training complete. Vocab size: {vocab_size}")

    def train_from_iterator(
        self,
        iterator,
        vocab_size: Optional[int] = None,
        length: Optional[int] = None,
    ) -> None:
        """Train the tokenizer from an iterator.

        Args:
            iterator: Iterator yielding text strings.
            vocab_size: Override the default vocabulary size.
            length: Optional length of the iterator for progress bar.
        """
        vocab_size = vocab_size or self.vocab_size

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=list(self.special_tokens.keys()),
            show_progress=True,
        )

        logger.info("Training tokenizer from iterator...")
        self.tokenizer.train_from_iterator(
            iterator=iterator,
            trainer=trainer,
            length=length,
        )
        logger.info(f"Tokenizer training complete. Vocab size: {vocab_size}")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token IDs.
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Encode a batch of texts.

        Args:
            texts: List of input texts.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token ID lists.
        """
        encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=add_special_tokens,
        )
        return [encoding.ids for encoding in encodings]

    def decode_batch(
        self,
        ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID lists.

        Args:
            ids_list: List of token ID lists.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded texts.
        """
        return self.tokenizer.decode_batch(
            ids_list,
            skip_special_tokens=skip_special_tokens,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer to disk.

        Args:
            path: Path to save directory or file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save(str(path))
        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CodeTokenizer":
        """Load a tokenizer from disk.

        Args:
            path: Path to tokenizer file.

        Returns:
            Loaded CodeTokenizer instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        # Create instance
        instance = cls.__new__(cls)
        instance.tokenizer = Tokenizer.from_file(str(path))
        instance.vocab_size = instance.tokenizer.get_vocab_size()

        # Reconstruct special tokens
        vocab = instance.tokenizer.get_vocab()
        instance.special_tokens = {
            token: idx
            for token, idx in vocab.items()
            if token.startswith("<|") and token.endswith("|>")
        }

        logger.info(f"Tokenizer loaded from {path}")
        return instance

    def get_vocab_size(self) -> int:
        """Get the vocabulary size.

        Returns:
            Vocabulary size.
        """
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary.

        Returns:
            Dictionary mapping tokens to IDs.
        """
        return self.tokenizer.get_vocab()

    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the padding token ID."""
        return self.special_tokens.get("<|pad|>")

    @property
    def bos_token_id(self) -> Optional[int]:
        """Get the beginning-of-sequence token ID."""
        return self.special_tokens.get("<|bos|>")

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get the end-of-sequence token ID."""
        return self.special_tokens.get("<|eos|>")
