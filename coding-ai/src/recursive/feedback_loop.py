"""Recursive feedback loop for self-improvement.

This module orchestrates the recursive learning process where the model:
1. Generates code samples
2. Evaluates them for correctness and quality
3. Adds successful samples to the training dataset
4. Periodically fine-tunes on new data
"""

import logging
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..data.dataset import DynamicDatasetManager, create_dataloader
from .evaluator import CodeEvaluator
from .generator import CodeGenerationTask

logger = logging.getLogger(__name__)


class RecursiveFeedbackLoop:
    """Manages the recursive self-improvement loop.

    This class coordinates code generation, evaluation, and dataset
    augmentation for continuous model improvement.

    Attributes:
        generation_task: Code generation task manager.
        evaluator: Code evaluator.
        dynamic_dataset: Dynamic dataset manager.
        config: Configuration dictionary.
    """

    def __init__(
        self,
        generation_task: CodeGenerationTask,
        evaluator: CodeEvaluator,
        dynamic_dataset: DynamicDatasetManager,
        config: Dict,
    ):
        """Initialize recursive feedback loop.

        Args:
            generation_task: Code generation task manager.
            evaluator: Code evaluator.
            dynamic_dataset: Dynamic dataset manager.
            config: Configuration dictionary.
        """
        self.generation_task = generation_task
        self.evaluator = evaluator
        self.dynamic_dataset = dynamic_dataset
        self.config = config

        # Statistics
        self.total_generated = 0
        self.total_valid = 0
        self.total_added = 0
        self.iteration = 0

        logger.info("RecursiveFeedbackLoop initialized")

    def run_iteration(
        self,
        num_samples: int,
        success_threshold: float = 0.7,
    ) -> Dict[str, any]:
        """Run one iteration of the recursive loop.

        Args:
            num_samples: Number of samples to generate.
            success_threshold: Minimum quality score to accept.

        Returns:
            Dictionary with iteration statistics.
        """
        self.iteration += 1
        logger.info(f"Starting recursive iteration {self.iteration}")

        start_time = time.time()

        # Step 1: Generate code samples
        logger.info(f"Generating {num_samples} code samples...")
        samples = self.generation_task.generate_samples(
            num_samples=num_samples,
            max_length=self.config.get("max_length", 256),
            temperature=self.config.get("temperature", 0.8),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 50),
        )

        self.total_generated += len(samples)

        # Step 2: Evaluate generated code
        logger.info(f"Evaluating {len(samples)} samples...")
        codes = [s["code"] for s in samples]
        eval_results = self.evaluator.evaluate_batch(codes)

        # Step 3: Filter valid samples
        logger.info("Filtering valid samples...")
        valid_results = self.evaluator.filter_valid_samples(
            eval_results,
            min_quality_score=success_threshold,
            require_execution_success=False,  # Can be configurable
        )

        self.total_valid += len(valid_results)

        # Step 4: Add to dynamic dataset
        logger.info(f"Adding {len(valid_results)} samples to dataset...")
        for i, result in enumerate(valid_results):
            metadata = {
                "iteration": self.iteration,
                "quality_score": result["quality_score"],
                "metrics": result["metrics"],
                "execution_success": result["execution_success"],
                "prompt": samples[i]["prompt"],
            }

            self.dynamic_dataset.add_sample(
                code=result["code"],
                metadata=metadata,
            )

        self.total_added += len(valid_results)

        # Save dynamic dataset
        self.dynamic_dataset.save()

        # Calculate statistics
        duration = time.time() - start_time
        success_rate = len(valid_results) / len(samples) if samples else 0.0

        stats = {
            "iteration": self.iteration,
            "num_generated": len(samples),
            "num_valid": len(valid_results),
            "success_rate": success_rate,
            "total_generated": self.total_generated,
            "total_valid": self.total_valid,
            "total_added": self.total_added,
            "dataset_size": len(self.dynamic_dataset.samples),
            "dataset_version": self.dynamic_dataset.version,
            "duration_seconds": duration,
        }

        # Calculate average quality metrics
        if valid_results:
            avg_quality = sum(r["quality_score"] for r in valid_results) / len(valid_results)
            avg_lines = sum(
                r["metrics"].get("num_lines", 0) for r in valid_results
            ) / len(valid_results)

            stats["avg_quality_score"] = avg_quality
            stats["avg_lines"] = avg_lines

        logger.info(
            f"Iteration {self.iteration} complete: "
            f"{len(valid_results)}/{len(samples)} samples added "
            f"(success_rate={success_rate:.2%})"
        )

        return stats

    def should_finetune(self, global_step: int, finetune_interval: int) -> bool:
        """Check if model should be fine-tuned on new data.

        Args:
            global_step: Current training step.
            finetune_interval: Fine-tune every N steps.

        Returns:
            True if fine-tuning should occur.
        """
        return global_step > 0 and global_step % finetune_interval == 0

    def get_statistics(self) -> Dict[str, any]:
        """Get overall statistics.

        Returns:
            Dictionary with statistics.
        """
        overall_success_rate = (
            self.total_valid / self.total_generated
            if self.total_generated > 0
            else 0.0
        )

        return {
            "total_iterations": self.iteration,
            "total_generated": self.total_generated,
            "total_valid": self.total_valid,
            "total_added": self.total_added,
            "overall_success_rate": overall_success_rate,
            "dataset_size": len(self.dynamic_dataset.samples),
            "dataset_version": self.dynamic_dataset.version,
            "dataset_stats": self.dynamic_dataset.get_stats(),
        }


class RecursiveTrainingIntegration:
    """Integrates recursive learning with main training loop.

    This class provides methods to integrate the recursive feedback loop
    with the main training process.
    """

    def __init__(
        self,
        feedback_loop: RecursiveFeedbackLoop,
        tokenizer,
        device: torch.device,
    ):
        """Initialize recursive training integration.

        Args:
            feedback_loop: RecursiveFeedbackLoop instance.
            tokenizer: Tokenizer instance.
            device: Device to train on.
        """
        self.feedback_loop = feedback_loop
        self.tokenizer = tokenizer
        self.device = device

        logger.info("RecursiveTrainingIntegration initialized")

    def should_run_iteration(
        self,
        global_step: int,
        start_after_steps: int,
        generation_interval: int,
    ) -> bool:
        """Check if a recursive iteration should run.

        Args:
            global_step: Current training step.
            start_after_steps: Start recursive loop after N steps.
            generation_interval: Run every N steps after starting.

        Returns:
            True if iteration should run.
        """
        if global_step < start_after_steps:
            return False

        steps_since_start = global_step - start_after_steps
        return steps_since_start % generation_interval == 0

    def run_and_log(
        self,
        num_samples: int,
        success_threshold: float,
        wandb_tracker: Optional[any] = None,
    ) -> Dict[str, any]:
        """Run recursive iteration and log results.

        Args:
            num_samples: Number of samples to generate.
            success_threshold: Quality threshold for acceptance.
            wandb_tracker: Optional W&B tracker for logging.

        Returns:
            Iteration statistics.
        """
        stats = self.feedback_loop.run_iteration(
            num_samples=num_samples,
            success_threshold=success_threshold,
        )

        # Log to wandb if available
        if wandb_tracker:
            try:
                wandb_metrics = {
                    f"recursive/{k}": v
                    for k, v in stats.items()
                    if isinstance(v, (int, float))
                }
                wandb_tracker.log(wandb_metrics)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        return stats

    def create_finetune_dataloader(
        self,
        batch_size: int,
        max_length: int = 512,
        version: Optional[int] = None,
    ) -> Optional[DataLoader]:
        """Create dataloader for fine-tuning on new samples.

        Args:
            batch_size: Batch size for dataloader.
            max_length: Maximum sequence length.
            version: Dataset version to use (None for all).

        Returns:
            DataLoader or None if no samples available.
        """
        samples = self.feedback_loop.dynamic_dataset.get_samples(version=version)

        if not samples:
            logger.warning("No samples available for fine-tuning")
            return None

        logger.info(f"Creating fine-tune dataloader with {len(samples)} samples")

        dataset = self.feedback_loop.dynamic_dataset.get_dataset(
            tokenizer=self.tokenizer,
            max_length=max_length,
            version=version,
        )

        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True,
        )

        return dataloader

    def finetune_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        num_epochs: int = 1,
    ) -> float:
        """Perform fine-tuning on new samples.

        Args:
            model: Model to fine-tune.
            optimizer: Optimizer to use.
            dataloader: Dataloader with new samples.
            num_epochs: Number of fine-tuning epochs.

        Returns:
            Average fine-tuning loss.
        """
        logger.info(f"Fine-tuning for {num_epochs} epochs...")

        model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs["loss"]

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Fine-tuning complete. Average loss: {avg_loss:.4f}")

        return avg_loss
