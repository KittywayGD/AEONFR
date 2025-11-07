"""Checkpoint management for robust training with pause/resume.

This module handles saving and loading of complete training state including
model parameters, optimizer state, scheduler state, RNG states, and training
metadata.
"""

import json
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing for training with automatic pause/resume.

    This class handles:
    - Periodic checkpoint saving
    - Time-based checkpoint saving
    - Signal handling for graceful interruption
    - Keeping only the N most recent checkpoints
    - Complete state restoration

    Attributes:
        save_dir: Directory to save checkpoints.
        save_steps: Save checkpoint every N steps.
        save_time_interval: Save checkpoint every N seconds.
        keep_last_n: Keep only the N most recent checkpoints.
        resume_from_checkpoint: Whether to automatically resume from latest.
    """

    def __init__(
        self,
        save_dir: str,
        save_steps: int = 500,
        save_time_interval: Optional[int] = None,
        keep_last_n: int = 3,
        resume_from_checkpoint: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints.
            save_steps: Save checkpoint every N training steps.
            save_time_interval: Save checkpoint every N seconds (optional).
            keep_last_n: Keep only the N most recent checkpoints.
            resume_from_checkpoint: Whether to auto-resume from latest checkpoint.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_steps = save_steps
        self.save_time_interval = save_time_interval
        self.keep_last_n = keep_last_n
        self.resume_from_checkpoint = resume_from_checkpoint

        self.last_save_time = time.time()
        self.should_stop = False

        # Set up signal handlers for graceful interruption
        self._setup_signal_handlers()

        logger.info(
            f"CheckpointManager initialized: save_dir={save_dir}, "
            f"save_steps={save_steps}, save_time_interval={save_time_interval}s, "
            f"keep_last_n={keep_last_n}"
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful interruption."""
        def signal_handler(signum, frame):
            logger.warning(
                f"Received signal {signum}. Setting flag for graceful shutdown..."
            )
            self.should_stop = True

        # Handle Ctrl+C (SIGINT) and termination (SIGTERM)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def should_save(self, global_step: int) -> bool:
        """Check if a checkpoint should be saved.

        Args:
            global_step: Current training step.

        Returns:
            True if checkpoint should be saved.
        """
        # Check step-based saving
        step_based = (global_step > 0) and (global_step % self.save_steps == 0)

        # Check time-based saving
        time_based = False
        if self.save_time_interval is not None:
            current_time = time.time()
            time_elapsed = current_time - self.last_save_time
            time_based = time_elapsed >= self.save_time_interval

        return step_based or time_based

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        global_step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a complete training checkpoint.

        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            scheduler: The learning rate scheduler to save (optional).
            global_step: Current training step.
            epoch: Current training epoch.
            metrics: Dictionary of metrics to save.
            extra_state: Additional state to save.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_name = f"checkpoint-{global_step}"
        checkpoint_path = self.save_dir / checkpoint_name

        # Create checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint data
        checkpoint_data = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": {
                "python": None,  # Python random state
                "numpy": None,   # NumPy random state
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "metrics": metrics or {},
            "extra_state": extra_state or {},
        }

        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Add Python and NumPy RNG states
        try:
            import random
            checkpoint_data["rng_state"]["python"] = random.getstate()
        except Exception as e:
            logger.warning(f"Failed to save Python RNG state: {e}")

        try:
            import numpy as np
            checkpoint_data["rng_state"]["numpy"] = np.random.get_state()
        except Exception as e:
            logger.warning(f"Failed to save NumPy RNG state: {e}")

        # Save checkpoint
        try:
            # Save model and optimizer in separate files for easier debugging
            torch.save(
                checkpoint_data["model_state_dict"],
                checkpoint_path / "model.pt",
            )
            torch.save(
                checkpoint_data["optimizer_state_dict"],
                checkpoint_path / "optimizer.pt",
            )

            if scheduler is not None:
                torch.save(
                    checkpoint_data["scheduler_state_dict"],
                    checkpoint_path / "scheduler.pt",
                )

            # Save RNG states
            torch.save(
                checkpoint_data["rng_state"],
                checkpoint_path / "rng_state.pt",
            )

            # Save metadata as JSON
            metadata = {
                "global_step": global_step,
                "epoch": epoch,
                "metrics": metrics or {},
                "extra_state": extra_state or {},
            }
            with open(checkpoint_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Checkpoint saved to {checkpoint_path}")
            self.last_save_time = time.time()

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        strict: bool = True,
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Load a training checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into (optional).
            checkpoint_path: Specific checkpoint to load (if None, loads latest).
            strict: Whether to strictly enforce state dict loading.

        Returns:
            Tuple of (global_step, epoch, extra_state).

        Raises:
            FileNotFoundError: If no checkpoint is found.
        """
        # Find checkpoint to load
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found to resume from")

        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            # Load model state
            model_state = torch.load(
                checkpoint_path / "model.pt",
                map_location="cpu",
            )
            model.load_state_dict(model_state, strict=strict)

            # Load optimizer state
            optimizer_state = torch.load(
                checkpoint_path / "optimizer.pt",
                map_location="cpu",
            )
            optimizer.load_state_dict(optimizer_state)

            # Load scheduler state if exists
            if scheduler is not None:
                scheduler_path = checkpoint_path / "scheduler.pt"
                if scheduler_path.exists():
                    scheduler_state = torch.load(scheduler_path, map_location="cpu")
                    scheduler.load_state_dict(scheduler_state)

            # Load RNG states
            rng_state_path = checkpoint_path / "rng_state.pt"
            if rng_state_path.exists():
                rng_state = torch.load(rng_state_path, map_location="cpu")
                self._restore_rng_states(rng_state)

            # Load metadata
            with open(checkpoint_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            global_step = metadata["global_step"]
            epoch = metadata["epoch"]
            extra_state = metadata.get("extra_state", {})

            logger.info(
                f"Checkpoint loaded: step={global_step}, epoch={epoch}"
            )

            return global_step, epoch, extra_state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _restore_rng_states(self, rng_state: Dict[str, Any]) -> None:
        """Restore random number generator states.

        Args:
            rng_state: Dictionary of RNG states.
        """
        # Restore PyTorch RNG state
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"])

        if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        # Restore Python RNG state
        if rng_state.get("python") is not None:
            try:
                import random
                random.setstate(rng_state["python"])
            except Exception as e:
                logger.warning(f"Failed to restore Python RNG state: {e}")

        # Restore NumPy RNG state
        if rng_state.get("numpy") is not None:
            try:
                import numpy as np
                np.random.set_state(rng_state["numpy"])
            except Exception as e:
                logger.warning(f"Failed to restore NumPy RNG state: {e}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]

    def _list_checkpoints(self) -> List[Path]:
        """List all checkpoints in the save directory.

        Returns:
            List of checkpoint paths sorted by step number.
        """
        if not self.save_dir.exists():
            return []

        checkpoints = [
            d for d in self.save_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]

        # Sort by step number
        def get_step(path: Path) -> int:
            try:
                return int(path.name.split("-")[1])
            except (IndexError, ValueError):
                return -1

        checkpoints.sort(key=get_step)
        return checkpoints

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoints = self._list_checkpoints()

        if len(checkpoints) <= self.keep_last_n:
            return

        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.keep_last_n]
        for checkpoint_path in to_remove:
            try:
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")

    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        global_step: int,
        epoch: int,
        metric_value: float,
        metric_name: str = "loss",
        mode: str = "min",
    ) -> Optional[Path]:
        """Save checkpoint if it's the best so far.

        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            scheduler: The learning rate scheduler to save.
            global_step: Current training step.
            epoch: Current training epoch.
            metric_value: Value of the metric to compare.
            metric_name: Name of the metric.
            mode: "min" for metrics to minimize, "max" for metrics to maximize.

        Returns:
            Path to saved checkpoint if it's the best, None otherwise.
        """
        best_checkpoint_path = self.save_dir / "best_checkpoint"
        best_metric_file = self.save_dir / "best_metric.json"

        # Load previous best metric
        is_best = True
        if best_metric_file.exists():
            with open(best_metric_file, "r") as f:
                best_metric_data = json.load(f)
                prev_best = best_metric_data.get("value", float("inf") if mode == "min" else float("-inf"))

                if mode == "min":
                    is_best = metric_value < prev_best
                else:
                    is_best = metric_value > prev_best

        if is_best:
            # Remove old best checkpoint
            if best_checkpoint_path.exists():
                shutil.rmtree(best_checkpoint_path)

            # Save new best checkpoint
            checkpoint_path = self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch,
                metrics={metric_name: metric_value},
            )

            # Copy to best_checkpoint directory
            shutil.copytree(checkpoint_path, best_checkpoint_path)

            # Save best metric info
            with open(best_metric_file, "w") as f:
                json.dump({
                    "value": metric_value,
                    "metric_name": metric_name,
                    "global_step": global_step,
                    "epoch": epoch,
                }, f, indent=2)

            logger.info(
                f"New best checkpoint saved with {metric_name}={metric_value}"
            )
            return best_checkpoint_path

        return None
