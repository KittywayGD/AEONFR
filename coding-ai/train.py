#!/usr/bin/env python3
"""Main training script for Recursive Code LLM.

This script orchestrates the complete training pipeline including:
- Model initialization
- Dataset preparation
- Training with checkpointing
- Recursive self-improvement loop
- Monitoring and logging
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import (
    CodeDataset,
    DynamicDatasetManager,
    StreamingCodeDataset,
    create_dataloader,
)
from model.architecture import ModelConfig, RecursiveCodeLLM
from model.tokenizer import CodeTokenizer
from recursive.evaluator import CodeEvaluator
from recursive.feedback_loop import RecursiveFeedbackLoop, RecursiveTrainingIntegration
from recursive.generator import CodeGenerationTask, CodeGenerator, PromptGenerator
from training.checkpoint import CheckpointManager
from training.trainer import Trainer


def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_dir: Directory for log files.
        log_level: Logging level.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info("Logging configured")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Configuration loaded from {config_path}")
    return config


def initialize_tokenizer(config: dict, tokenizer_path: Optional[str] = None) -> CodeTokenizer:
    """Initialize or load tokenizer.

    Args:
        config: Configuration dictionary.
        tokenizer_path: Path to existing tokenizer (optional).

    Returns:
        CodeTokenizer instance.
    """
    if tokenizer_path and Path(tokenizer_path).exists():
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = CodeTokenizer.load(tokenizer_path)
    else:
        logging.info("Creating new tokenizer")
        tokenizer_config = config["tokenizer"]
        tokenizer = CodeTokenizer(
            vocab_size=tokenizer_config["vocab_size"],
            special_tokens=tokenizer_config["special_tokens"],
            min_frequency=tokenizer_config.get("min_frequency", 2),
        )

        # Train tokenizer if dataset available
        dataset_config = config["dataset"]
        if dataset_config.get("use_huggingface_dataset"):
            logging.info("Training tokenizer on HuggingFace dataset...")
            from datasets import load_dataset

            dataset = load_dataset(
                dataset_config["hf_dataset_name"],
                dataset_config["hf_dataset_config"],
                split="train",
                streaming=True,
            )

            # Get iterator of text samples
            def text_iterator():
                for i, sample in enumerate(dataset):
                    if i >= 10000:  # Train on first 10k samples
                        break
                    if "content" in sample:
                        yield sample["content"]
                    elif "text" in sample:
                        yield sample["text"]
                    elif "code" in sample:
                        yield sample["code"]

            tokenizer.train_from_iterator(
                text_iterator(),
                vocab_size=tokenizer_config["vocab_size"],
            )

            # Save tokenizer
            tokenizer_save_path = Path("./checkpoints/tokenizer.json")
            tokenizer.save(tokenizer_save_path)
            logging.info(f"Tokenizer saved to {tokenizer_save_path}")

    return tokenizer


def initialize_model(config: dict, tokenizer: CodeTokenizer) -> RecursiveCodeLLM:
    """Initialize model.

    Args:
        config: Configuration dictionary.
        tokenizer: Tokenizer instance.

    Returns:
        RecursiveCodeLLM model.
    """
    model_config = config["model"]

    # Create model config
    config_obj = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=model_config["hidden_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        intermediate_size=model_config["intermediate_size"],
        max_position_embeddings=model_config["max_position_embeddings"],
        dropout=model_config["dropout"],
        attention_dropout=model_config["attention_dropout"],
        layer_norm_eps=model_config["layer_norm_eps"],
        initializer_range=model_config["initializer_range"],
        use_flash_attention=model_config.get("use_flash_attention", False),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
    )

    # Create model
    model = RecursiveCodeLLM(config_obj)
    logging.info(f"Model initialized with {model.num_parameters():,} parameters")

    return model


def create_dataloaders(
    config: dict,
    tokenizer: CodeTokenizer,
) -> tuple:
    """Create training and evaluation dataloaders.

    Args:
        config: Configuration dictionary.
        tokenizer: Tokenizer instance.

    Returns:
        Tuple of (train_dataloader, eval_dataloader).
    """
    dataset_config = config["dataset"]
    training_config = config["training"]

    # Create training dataloader
    if dataset_config.get("use_huggingface_dataset"):
        train_dataset = StreamingCodeDataset(
            dataset_name=dataset_config["hf_dataset_name"],
            dataset_config=dataset_config["hf_dataset_config"],
            tokenizer=tokenizer,
            max_length=dataset_config["max_length"],
            split="train",
            streaming=True,
        )
    else:
        # Load from file (implement as needed)
        raise NotImplementedError("File-based dataset loading not yet implemented")

    train_dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=training_config["batch_size"],
        num_workers=dataset_config.get("num_workers", 4),
        shuffle=False,  # Streaming datasets don't support shuffle
    )

    # Eval dataloader (optional)
    eval_dataloader = None
    # Implement eval dataset if needed

    logging.info("Dataloaders created")
    return train_dataloader, eval_dataloader


def initialize_recursive_components(
    model: RecursiveCodeLLM,
    tokenizer: CodeTokenizer,
    config: dict,
    device: torch.device,
) -> tuple:
    """Initialize components for recursive learning.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        config: Configuration dictionary.
        device: Device to use.

    Returns:
        Tuple of (feedback_loop, integration).
    """
    recursive_config = config["recursive"]

    # Initialize components
    code_generator = CodeGenerator(model, tokenizer, device)
    prompt_generator = PromptGenerator(seed=config["training"]["seed"])
    generation_task = CodeGenerationTask(code_generator, prompt_generator)

    evaluator = CodeEvaluator(
        timeout=recursive_config["evaluation"]["sandbox_timeout"],
        use_docker=recursive_config["evaluation"]["use_docker"],
        docker_image=recursive_config["evaluation"].get("docker_image", "python:3.10-slim"),
    )

    dynamic_dataset = DynamicDatasetManager(
        dataset_path=config["dataset"]["dynamic_dataset_path"],
        max_samples=config["dataset"]["max_dynamic_samples"],
        version=config["dataset"].get("dynamic_dataset_version", 1),
    )

    feedback_loop = RecursiveFeedbackLoop(
        generation_task=generation_task,
        evaluator=evaluator,
        dynamic_dataset=dynamic_dataset,
        config=recursive_config["generation"],
    )

    integration = RecursiveTrainingIntegration(
        feedback_loop=feedback_loop,
        tokenizer=tokenizer,
        device=device,
    )

    logging.info("Recursive learning components initialized")
    return feedback_loop, integration


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Recursive Code LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to pre-trained tokenizer",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        log_dir=config["logging"]["log_dir"],
        log_level=config["logging"]["log_level"],
    )

    logging.info("=" * 80)
    logging.info("Starting Recursive Code LLM Training")
    logging.info("=" * 80)

    # Set random seed
    set_seed(config["training"]["seed"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(config, args.tokenizer)

    # Initialize model
    model = initialize_model(config, tokenizer)
    model = model.to(device)

    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(config, tokenizer)

    # Initialize optimizer and scheduler
    training_config = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # Calculate total steps
    num_epochs = training_config["num_epochs"]
    # For streaming datasets, estimate steps per epoch
    steps_per_epoch = 10000  # Adjust based on dataset size
    total_steps = num_epochs * steps_per_epoch

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=training_config["learning_rate"] * 0.1,
    )

    # Initialize checkpoint manager
    checkpoint_config = config["checkpoint"]
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_config["save_dir"],
        save_steps=checkpoint_config["save_steps"],
        save_time_interval=checkpoint_config.get("save_time_interval"),
        keep_last_n=checkpoint_config["keep_last_n"],
        resume_from_checkpoint=args.resume or checkpoint_config["resume_from_checkpoint"],
    )

    # Initialize wandb if enabled
    wandb_tracker = None
    if config["logging"].get("use_wandb", False):
        try:
            import wandb

            wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity"),
                name=config["logging"].get("wandb_run_name"),
                config=config,
            )
            wandb_tracker = wandb
            logging.info("W&B tracking initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize W&B: {e}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config,
        checkpoint_manager=checkpoint_manager,
        wandb_tracker=wandb_tracker,
    )

    # Initialize recursive learning components if enabled
    recursive_integration = None
    if config["recursive"].get("enabled", False):
        _, recursive_integration = initialize_recursive_components(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )

    # Training loop with recursive learning
    logging.info("Starting training...")

    try:
        # Run training
        trainer.train(
            num_epochs=num_epochs,
            max_grad_norm=training_config["max_grad_norm"],
            eval_steps=config["evaluation"].get("eval_steps"),
            log_interval=config["logging"]["log_interval"],
        )

        logging.info("Training completed successfully!")

        # Save final model
        trainer.save_model("./checkpoints/final_model")

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        if wandb_tracker:
            wandb_tracker.finish()

    logging.info("=" * 80)
    logging.info("Training session ended")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
