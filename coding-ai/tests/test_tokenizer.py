"""Tests for CodeTokenizer."""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.tokenizer import CodeTokenizer


class TestCodeTokenizer:
    """Test cases for CodeTokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = CodeTokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
        assert len(tokenizer.special_tokens) > 0

    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = CodeTokenizer(vocab_size=1000)

        # Train on a small sample
        sample_code = [
            "def hello():\n    print('Hello, World!')",
            "def add(x, y):\n    return x + y",
            "class MyClass:\n    pass",
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            for code in sample_code:
                f.write(code + "\n")
            temp_file = f.name

        try:
            tokenizer.train([temp_file])

            # Test encoding
            text = "def test():"
            encoded = tokenizer.encode(text)
            assert isinstance(encoded, list)
            assert len(encoded) > 0

            # Test decoding
            decoded = tokenizer.decode(encoded)
            assert isinstance(decoded, str)

        finally:
            Path(temp_file).unlink()

    def test_special_tokens(self):
        """Test special tokens."""
        special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>"]
        tokenizer = CodeTokenizer(special_tokens=special_tokens)

        assert tokenizer.pad_token_id is not None
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

    def test_save_load(self):
        """Test saving and loading tokenizer."""
        tokenizer = CodeTokenizer(vocab_size=1000)

        # Train on sample
        sample_code = ["def hello(): pass"]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_code[0])
            temp_file = f.name

        try:
            tokenizer.train([temp_file])

            # Save tokenizer
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "tokenizer.json"
                tokenizer.save(save_path)

                # Load tokenizer
                loaded_tokenizer = CodeTokenizer.load(save_path)

                assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

                # Test that loaded tokenizer works
                text = "def test():"
                encoded1 = tokenizer.encode(text)
                encoded2 = loaded_tokenizer.encode(text)
                assert encoded1 == encoded2

        finally:
            Path(temp_file).unlink()

    def test_batch_encoding(self):
        """Test batch encoding."""
        tokenizer = CodeTokenizer(vocab_size=1000)

        # Train
        sample_code = ["def hello(): pass", "x = 42"]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            for code in sample_code:
                f.write(code + "\n")
            temp_file = f.name

        try:
            tokenizer.train([temp_file])

            # Batch encode
            texts = ["def test1():", "def test2():"]
            encoded_batch = tokenizer.encode_batch(texts)

            assert len(encoded_batch) == len(texts)
            assert all(isinstance(ids, list) for ids in encoded_batch)

        finally:
            Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
