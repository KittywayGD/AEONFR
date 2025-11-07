"""Training loop with Accelerate for distributed and optimized training.

This module implements the main training loop with support for:
- Mixed precision training
- Gradient accumulation
- Distributed training
- Checkpoint management
- Logging and monitoring
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for code LLM with Accelerate.

    This class handles the complete training loop with checkpointing,
    logging, and evaluation.

    Attributes:
        model: The model to train.
        optimizer: The optimizer.
        scheduler: Learning rate scheduler.
        train_dataloader: Training data loader.
        eval_dataloader: Evaluation data loader (optional).
        accelerator: Accelerate accelerator instance.
        checkpoint_manager: Checkpoint manager.
        config: Training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: Dict,
        checkpoint_manager: CheckpointManager,
        wandb_tracker=None,
    ):
        """Initialize the trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            train_dataloader: Training data loader.
            eval_dataloader: Optional evaluation data loader.
            config: Training configuration dictionary.
            checkpoint_manager: Checkpoint manager instance.
            wandb_tracker: Optional W&B tracker.
        """
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.wandb_tracker = wandb_tracker

        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "no"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            log_with="wandb" if wandb_tracker else None,
        )

        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
            train_dataloader,
        )

        if eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)
        else:
            self.eval_dataloader = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float("inf")

        logger.info(
            f"Trainer initialized on device: {self.accelerator.device}, "
            f"mixed_precision: {config.get('mixed_precision', 'no')}"
        )

    def train(
        self,
        num_epochs: int,
        max_grad_norm: float = 1.0,
        eval_steps: Optional[int] = None,
        log_interval: int = 10,
    ) -> None:
        """Run the training loop.

        Args:
            num_epochs: Number of epochs to train.
            max_grad_norm: Maximum gradient norm for clipping.
            eval_steps: Run evaluation every N steps (None to skip).
            log_interval: Log metrics every N steps.
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Try to resume from checkpoint
        if self.checkpoint_manager.resume_from_checkpoint:
            try:
                self.global_step, self.current_epoch, _ = \
                    self.checkpoint_manager.load_checkpoint(
                        self.accelerator.unwrap_model(self.model),
                        self.optimizer,
                        self.scheduler,
                    )
                logger.info(
                    f"Resumed from checkpoint: epoch={self.current_epoch}, "
                    f"step={self.global_step}"
                )
            except FileNotFoundError:
                logger.info("No checkpoint found, starting from scratch")

        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self._train_epoch(
                epoch=epoch,
                max_grad_norm=max_grad_norm,
                eval_steps=eval_steps,
                log_interval=log_interval,
            )

            # Check for interruption signal
            if self.checkpoint_manager.should_stop:
                logger.warning("Received stop signal, saving checkpoint and exiting...")
                self._save_checkpoint()
                break

        logger.info("Training completed!")

    def _train_epoch(
        self,
        epoch: int,
        max_grad_norm: float,
        eval_steps: Optional[int],
        log_interval: int,
    ) -> None:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.
            max_grad_norm: Maximum gradient norm for clipping.
            eval_steps: Evaluation interval in steps.
            log_interval: Logging interval in steps.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs["loss"]

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm,
                    )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update global step only when we actually update weights
            if self.accelerator.sync_gradients:
                self.global_step += 1

                # Logging
                if self.global_step % log_interval == 0:
                    avg_loss = total_loss / num_batches
                    perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

                    metrics = {
                        "train/loss": avg_loss,
                        "train/perplexity": perplexity,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": self.global_step,
                    }

                    self._log_metrics(metrics)

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{perplexity:.2f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    })

                    # Reset for next logging interval
                    total_loss = 0.0
                    num_batches = 0

                # Evaluation
                if eval_steps and self.global_step % eval_steps == 0:
                    if self.eval_dataloader is not None:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics)

                        # Save best checkpoint
                        if eval_metrics["eval/loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval/loss"]
                            self.checkpoint_manager.save_best_checkpoint(
                                model=self.accelerator.unwrap_model(self.model),
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                global_step=self.global_step,
                                epoch=epoch,
                                metric_value=self.best_eval_loss,
                                metric_name="eval_loss",
                                mode="min",
                            )

                        self.model.train()

                # Checkpointing
                if self.checkpoint_manager.should_save(self.global_step):
                    self._save_checkpoint()

                # Check for interruption
                if self.checkpoint_manager.should_stop:
                    break

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation set.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided")
            return {}

        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs["loss"]
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
            "eval/global_step": self.global_step,
        }

        logger.info(
            f"Evaluation results - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}"
        )

        return metrics

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        if not self.accelerator.is_local_main_process:
            return

        logger.info(f"Saving checkpoint at step {self.global_step}")

        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=unwrapped_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.current_epoch,
            metrics={
                "best_eval_loss": self.best_eval_loss,
            },
        )

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to console and tracking systems.

        Args:
            metrics: Dictionary of metrics to log.
        """
        if not self.accelerator.is_local_main_process:
            return

        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} - {metrics_str}")

        # Log to wandb if available
        if self.wandb_tracker:
            try:
                self.wandb_tracker.log(metrics, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

    def save_model(self, output_dir: str) -> None:
        """Save the trained model.

        Args:
            output_dir: Directory to save the model.
        """
        if not self.accelerator.is_local_main_process:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model state dict
        torch.save(
            unwrapped_model.state_dict(),
            output_dir / "pytorch_model.bin",
        )

        # Save model config
        if hasattr(unwrapped_model, "config"):
            import json
            config_dict = vars(unwrapped_model.config)
            with open(output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {output_dir}")

    def load_model(self, model_path: str) -> None:
        """Load a trained model.

        Args:
            model_path: Path to the model file.
        """
        logger.info(f"Loading model from {model_path}")

        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu")
        unwrapped_model.load_state_dict(state_dict)

        logger.info("Model loaded successfully")
