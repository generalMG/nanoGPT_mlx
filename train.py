"""
Training script for GPT model using MLX.

This module provides a complete training pipeline for GPT-style language models
using the MLX framework optimized for Apple Silicon.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from model import GPT, GPTConfig
from tboard_utils import get_tensorboard, init_tensorboard

# Set random seed for reproducibility
mx.random.seed(1337)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.

    Attributes:
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout rate.
        bias: Whether to use bias in linear layers.
        d_type: Data type for model parameters.
        learning_rate: Peak learning rate.
        min_lr: Minimum learning rate after decay.
        weight_decay: Weight decay coefficient.
        beta1: Adam beta1 parameter.
        beta2: Adam beta2 parameter.
        num_iters: Total number of training iterations.
        warmup_iters: Number of warmup iterations.
        lr_decay_iters: Number of iterations for LR decay.
        batch_size: Batch size per step.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        context_size: Context window size.
        dataset: Dataset name.
        meta_vocab_size: Optional vocabulary size override.
        out_dir: Output directory for checkpoints.
        save_interval: Save checkpoint every N iterations.
        eval_interval: Evaluate every N iterations.
        log_interval: Log metrics every N iterations.
        eval_only: Only run evaluation.
        resume: Path to checkpoint to resume from.
    """

    # Model architecture
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    d_type: str = "float32"

    # Optimizer settings
    learning_rate: float = 6.0e-4
    min_lr: float = 6.0e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    # Training settings
    num_iters: int = 600000
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    batch_size: int = 1
    gradient_accumulation_steps: int = 512
    context_size: int = 1024

    # Data settings
    dataset: str = "shakespeare"
    meta_vocab_size: Optional[int] = None

    # Logging and checkpointing
    out_dir: str = "gpt2_shakespeare_pretrain_mlx"
    save_interval: int = 1
    eval_interval: int = 10
    log_interval: int = 10
    eval_only: bool = False
    resume: Optional[str] = None

    def validate(self) -> None:
        """
        Validate training configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {self.n_embd}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {self.min_lr}")
        if self.min_lr > self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr}) cannot exceed learning_rate ({self.learning_rate})"
            )
        if self.num_iters <= 0:
            raise ValueError(f"num_iters must be positive, got {self.num_iters}")
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {self.warmup_iters}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
        if self.context_size <= 0:
            raise ValueError(f"context_size must be positive, got {self.context_size}")
        if self.d_type not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"d_type must be float32, float16, or bfloat16, got {self.d_type}")


def parse_args() -> TrainingConfig:
    """
    Parse command-line arguments into a TrainingConfig.

    Returns:
        TrainingConfig instance with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a GPT model with MLX")

    # Model architecture
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--bias",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Use bias in linear layers",
    )
    parser.add_argument(
        "--d_type", type=str, default="float32", help="Data type for model parameters"
    )

    # Optimizer settings
    parser.add_argument(
        "--learning_rate", type=float, default=6.0e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=6.0e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-1, help="Weight decay coefficient"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")

    # Training settings
    parser.add_argument(
        "--num_iters", type=int, default=600000, help="Total number of training iterations"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=2000, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=600000,
        help="Number of iterations for learning rate decay",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=512,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--context_size", type=int, default=1024, help="Context window size"
    )

    # Data settings
    parser.add_argument("--dataset", type=str, default="shakespeare", help="Dataset name")
    parser.add_argument(
        "--meta_vocab_size",
        type=int,
        default=None,
        help="Vocabulary size (None for default 50304)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--out_dir",
        type=str,
        default="gpt2_shakespeare_pretrain_mlx",
        help="Output directory",
    )
    parser.add_argument(
        "--save_interval", type=int, default=1, help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=10, help="Evaluate every N iterations"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log training metrics every N iterations"
    )
    parser.add_argument(
        "--eval_only",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Only run evaluation",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )

    args = parser.parse_args()

    return TrainingConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        d_type=args.d_type,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        num_iters=args.num_iters,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        context_size=args.context_size,
        dataset=args.dataset,
        meta_vocab_size=args.meta_vocab_size,
        out_dir=args.out_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_only=args.eval_only,
        resume=args.resume,
    )


def load_binary_data(file_path: str, dtype: str) -> mx.array:
    """
    Load binary token data from file.

    Args:
        file_path: Path to the binary file.
        dtype: Data type code for memoryview casting (e.g., 'H' for uint16).

    Returns:
        MLX array containing the loaded tokens.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            f"Please run 'python data/<dataset>/prepare.py' first."
        )
    with open(file_path, "rb") as f:
        data = f.read()
    return mx.array(memoryview(data).cast(dtype))


def get_batch(
    train_data: mx.array,
    val_data: mx.array,
    split: str,
    batch_size: int,
    context_size: int,
) -> Tuple[mx.array, mx.array]:
    """
    Get a random batch of training or validation data.

    Args:
        train_data: Training dataset as MLX array.
        val_data: Validation dataset as MLX array.
        split: Either 'train' or 'val'.
        batch_size: Number of sequences in the batch.
        context_size: Length of each sequence.

    Returns:
        Tuple of (input tokens, target tokens).
    """
    data = train_data if split == "train" else val_data
    ix = mx.random.randint(0, len(data) - context_size, shape=(batch_size,)).tolist()
    x = mx.stack([mx.array(data[i : i + context_size]) for i in ix]).astype(mx.int64)
    y = mx.stack([mx.array(data[i + 1 : i + 1 + context_size]) for i in ix]).astype(mx.int64)
    return x, y


def compute_learning_rate(
    iteration: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        iteration: Current training iteration.
        learning_rate: Peak learning rate.
        min_lr: Minimum learning rate.
        warmup_iters: Number of warmup iterations.
        lr_decay_iters: Total iterations for decay schedule.

    Returns:
        Learning rate for the current iteration.
    """
    # Linear warmup
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters

    # After decay period, return minimum
    if iteration > lr_decay_iters:
        return min_lr

    # Cosine decay
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def log_tensorboard_metrics(log_dict: Dict[str, float], iteration: int, prefix: str) -> None:
    """
    Log metrics to TensorBoard.

    Args:
        log_dict: Dictionary of metric names to values.
        iteration: Current training iteration.
        prefix: Prefix for metric names (e.g., 'train', 'val').
    """
    writer = get_tensorboard()
    for key, value in log_dict.items():
        writer.add_scalar(f"{prefix}/{key}", value, iteration)


def save_checkpoint(
    model: GPT,
    optimizer: optim.AdamW,
    iteration: int,
    config: TrainingConfig,
    model_config: GPTConfig,
    out_dir: str,
) -> None:
    """
    Save model checkpoint with optimizer state for resuming.

    Args:
        model: The GPT model.
        optimizer: The optimizer.
        iteration: Current training iteration.
        config: Training configuration.
        model_config: Model configuration.
        out_dir: Output directory.
    """
    # Save model weights
    weights_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}.npz")
    flat_params = tree_flatten(model.parameters())
    mx.savez(weights_path, **dict(flat_params))

    # Save model config
    config_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}.json")
    with open(config_path, "w") as f:
        json.dump(model_config.__dict__, f, indent=2)

    # Save training state for resuming
    training_state = {
        "iteration": iteration,
        "learning_rate": float(optimizer.learning_rate),
        "training_config": {
            "learning_rate": config.learning_rate,
            "min_lr": config.min_lr,
            "warmup_iters": config.warmup_iters,
            "lr_decay_iters": config.lr_decay_iters,
            "num_iters": config.num_iters,
        },
    }
    state_path = os.path.join(out_dir, "training_state.json")
    with open(state_path, "w") as f:
        json.dump(training_state, f, indent=2)

    logger.info(f"Saved checkpoint at iteration {iteration} to {out_dir}")


def load_checkpoint(
    checkpoint_dir: str,
) -> Tuple[Optional[Dict], Optional[Dict], int]:
    """
    Load checkpoint for resuming training.

    Args:
        checkpoint_dir: Directory containing the checkpoint.

    Returns:
        Tuple of (model_config, training_state, start_iteration).
        Returns (None, None, 0) if checkpoint doesn't exist.
    """
    base = os.path.basename(os.path.normpath(checkpoint_dir))
    config_path = os.path.join(checkpoint_dir, f"{base}.json")
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    weights_path = os.path.join(checkpoint_dir, f"{base}.npz")

    if not all(os.path.exists(p) for p in [config_path, weights_path]):
        logger.warning(f"Checkpoint not found in {checkpoint_dir}, starting from scratch")
        return None, None, 0

    with open(config_path, "r") as f:
        model_config = json.load(f)

    training_state = None
    start_iteration = 0
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            training_state = json.load(f)
            start_iteration = training_state.get("iteration", 0) + 1

    logger.info(f"Loaded checkpoint from {checkpoint_dir}, resuming from iteration {start_iteration}")
    return model_config, training_state, start_iteration


class Trainer:
    """
    Main trainer class for GPT model.

    Handles the training loop, gradient accumulation, checkpointing,
    and logging.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        config.validate()

        # Setup paths
        self.data_dir = os.path.join("data", config.dataset)
        self.out_dir = config.out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Initialize tensorboard
        tboard_dir = os.path.join(self.out_dir, "tboard_log")
        init_tensorboard(tboard_dir)

        # Load data
        logger.info(f"Loading data from {self.data_dir}")
        self.train_data = load_binary_data(os.path.join(self.data_dir, "train.bin"), "H")
        self.val_data = load_binary_data(os.path.join(self.data_dir, "val.bin"), "H")
        logger.info(
            f"Loaded {len(self.train_data):,} training tokens, "
            f"{len(self.val_data):,} validation tokens"
        )

        # Handle checkpoint resuming
        self.start_iteration = 0
        model_config_dict = None

        if config.resume:
            model_config_dict, training_state, self.start_iteration = load_checkpoint(
                config.resume
            )

        # Initialize model
        self.model, self.model_config = self._init_model(model_config_dict)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            betas=[config.beta1, config.beta2],
            weight_decay=config.weight_decay,
        )

        # Load weights if resuming
        if config.resume and model_config_dict is not None:
            base = os.path.basename(os.path.normpath(config.resume))
            weights_path = os.path.join(config.resume, f"{base}.npz")
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path, strict=False)
                logger.info(f"Loaded model weights from {weights_path}")

        # Setup loss function with gradient computation
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

    def _init_model(
        self, model_config_dict: Optional[Dict] = None
    ) -> Tuple[GPT, GPTConfig]:
        """
        Initialize the GPT model.

        Args:
            model_config_dict: Optional dict to override model config (for resuming).

        Returns:
            Tuple of (model, model_config).
        """
        if model_config_dict is not None:
            # Use config from checkpoint
            model_config = GPTConfig(**model_config_dict)
        else:
            # Create new config from training config
            vocab_size = self.config.meta_vocab_size
            if vocab_size is None:
                logger.info("Defaulting vocab_size to GPT-2's 50304 (50257 rounded up)")
                vocab_size = 50304

            model_config = GPTConfig(
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                n_embd=self.config.n_embd,
                block_size=self.config.context_size,
                bias=self.config.bias,
                vocab_size=vocab_size,
                dropout=self.config.dropout,
            )

        model = GPT(model_config)

        # Convert to specified dtype
        dtype = getattr(mx, self.config.d_type)
        weights = tree_map(lambda p: p.astype(dtype), model.parameters())
        model.update(weights)
        mx.eval(model.parameters())

        # Log model info
        nparams = sum(
            x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
        )
        logger.info(f"Model architecture:\n{model}")
        logger.info(f"Training a transformer with {nparams / 1024**2:.3f}M parameters")

        return model, model_config

    def _loss_fn(
        self, model: GPT, x: mx.array, y: mx.array, reduce: bool = True
    ) -> mx.array:
        """
        Compute cross-entropy loss.

        Args:
            model: The GPT model.
            x: Input token indices.
            y: Target token indices.
            reduce: Whether to reduce the loss to a scalar.

        Returns:
            Loss value.
        """
        logits = model(x)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    def _training_step(
        self, inputs: mx.array, targets: mx.array
    ) -> mx.array:
        """
        Perform a single training step with gradient accumulation.

        Args:
            inputs: Input token batch.
            targets: Target token batch.

        Returns:
            Average loss for the step.
        """
        accumulated_grads = tree_map(
            lambda x: mx.zeros_like(x), self.model.parameters()
        )
        accumulated_loss = 0.0

        for _ in range(self.config.gradient_accumulation_steps):
            loss, grads = self.loss_and_grad_fn(self.model, inputs, targets)

            accumulated_grads = tree_map(
                lambda acc, new: acc + new * (1.0 / self.config.gradient_accumulation_steps),
                accumulated_grads,
                grads,
            )

            tree_map(lambda grad: mx.eval(grad), accumulated_grads)
            accumulated_loss += loss.item()

        # Scale loss for logging
        loss = mx.array(accumulated_loss / self.config.gradient_accumulation_steps)

        # Apply gradients
        self.optimizer.update(self.model, accumulated_grads)

        return loss

    def train(self) -> None:
        """Run the main training loop."""
        if self.config.eval_only:
            logger.info("Eval-only mode, exiting")
            return

        # Get initial batch
        X, Y = get_batch(
            self.train_data,
            self.val_data,
            "train",
            self.config.batch_size,
            self.config.context_size,
        )

        state = [self.model.state, self.optimizer.state]
        tic = time.perf_counter()
        iter_num = self.start_iteration

        logger.info(f"Starting training from iteration {iter_num}")

        while iter_num <= self.config.num_iters:
            # Update learning rate
            new_lr = compute_learning_rate(
                iter_num,
                self.config.learning_rate,
                self.config.min_lr,
                self.config.warmup_iters,
                self.config.lr_decay_iters,
            )
            self.optimizer.learning_rate = new_lr

            # Training step
            loss = self._training_step(X, Y)

            # Prefetch next batch
            X, Y = get_batch(
                self.train_data,
                self.val_data,
                "train",
                self.config.batch_size,
                self.config.context_size,
            )

            # Logging
            toc = time.perf_counter()
            logger.info(
                f"iter {iter_num}: loss {loss.item():.4f}, "
                f"it/sec {1.0 / (toc - tic):.3f}, "
                f"lr {new_lr:.2e}"
            )
            tic = toc

            mx.eval(state)

            # TensorBoard logging
            if iter_num % self.config.log_interval == 0:
                log_tensorboard_metrics(
                    {"loss": loss.item(), "lr": new_lr},
                    iter_num,
                    "train",
                )

            # Save checkpoint
            if iter_num % self.config.save_interval == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    iter_num,
                    self.config,
                    self.model_config,
                    self.out_dir,
                )

            iter_num += 1

        logger.info("Training complete!")


def main() -> None:
    """Main entry point for training."""
    config = parse_args()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
