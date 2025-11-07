#!/usr/bin/env python3
"""Inference script for generating code with trained model.

This script loads a trained model and generates code from prompts.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model.architecture import RecursiveCodeLLM
from model.tokenizer import CodeTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    device: str = "cuda",
) -> tuple:
    """Load model and tokenizer.

    Args:
        model_path: Path to model checkpoint.
        tokenizer_path: Path to tokenizer.
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = CodeTokenizer.load(tokenizer_path)

    logger.info(f"Loading model from {model_path}")
    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)

    # Create model (you may need to load config separately)
    from model.architecture import ModelConfig

    # This is a simplified version - in practice, load config from file
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    model = RecursiveCodeLLM(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def generate_code(
    model: RecursiveCodeLLM,
    tokenizer: CodeTokenizer,
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    num_samples: int = 1,
) -> list:
    """Generate code from prompt.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        prompt: Input prompt.
        max_length: Maximum length to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        num_samples: Number of samples to generate.

    Returns:
        List of generated code strings.
    """
    logger.info(f"Generating {num_samples} samples from prompt: {prompt[:50]}...")

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_codes = []
    for ids in generated_ids:
        code = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        # Remove prompt from output
        if code.startswith(prompt):
            code = code[len(prompt):].strip()
        generated_codes.append(code)

    return generated_codes


def interactive_mode(model: RecursiveCodeLLM, tokenizer: CodeTokenizer) -> None:
    """Run in interactive mode.

    Args:
        model: The model.
        tokenizer: The tokenizer.
    """
    logger.info("Starting interactive mode. Type 'quit' to exit.")
    print("\n" + "=" * 80)
    print("Interactive Code Generation")
    print("=" * 80)
    print("Enter a prompt and the model will generate code.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80 + "\n")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not prompt:
                continue

            # Generate
            codes = generate_code(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=256,
                temperature=0.8,
                top_p=0.95,
                num_samples=1,
            )

            # Display
            print("\n" + "-" * 80)
            print("Generated Code:")
            print("-" * 80)
            print(prompt + codes[0])
            print("-" * 80)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error generating code: {e}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate code with trained model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for code generation (interactive if not provided)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    # Generate or interactive mode
    if args.prompt:
        # Single generation
        codes = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_samples=args.num_samples,
        )

        # Display results
        print("\n" + "=" * 80)
        print(f"Prompt: {args.prompt}")
        print("=" * 80)
        for i, code in enumerate(codes, 1):
            print(f"\nSample {i}:")
            print("-" * 80)
            print(args.prompt + code)
            print("-" * 80)

    else:
        # Interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
