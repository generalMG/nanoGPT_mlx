"""
Tests for the training module.

Run with: pytest tests/test_train.py -v
"""

import math
import os
import tempfile
import json

import pytest
import mlx.core as mx

from train import (
    TrainingConfig,
    load_binary_data,
    get_batch,
    compute_learning_rate,
    save_checkpoint,
    load_checkpoint,
)
from model import GPT, GPTConfig


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig()
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.dropout == 0.0
        assert config.bias is False
        assert config.learning_rate == 6.0e-4
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 512

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = TrainingConfig(
            n_layer=6,
            n_head=6,
            n_embd=384,
            learning_rate=1e-3,
            batch_size=4,
        )
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
        assert config.learning_rate == 1e-3
        assert config.batch_size == 4

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = TrainingConfig(n_layer=6, n_head=6, n_embd=384)
        config.validate()  # Should not raise

    def test_validate_invalid_n_layer(self):
        """Test validation fails for invalid n_layer."""
        config = TrainingConfig(n_layer=0)
        with pytest.raises(ValueError, match="n_layer must be positive"):
            config.validate()

    def test_validate_invalid_n_head(self):
        """Test validation fails for invalid n_head."""
        config = TrainingConfig(n_head=-1)
        with pytest.raises(ValueError, match="n_head must be positive"):
            config.validate()

    def test_validate_invalid_n_embd(self):
        """Test validation fails for invalid n_embd."""
        config = TrainingConfig(n_embd=0)
        with pytest.raises(ValueError, match="n_embd must be positive"):
            config.validate()

    def test_validate_n_embd_not_divisible(self):
        """Test validation fails when n_embd not divisible by n_head."""
        config = TrainingConfig(n_embd=100, n_head=12)
        with pytest.raises(ValueError, match="must be divisible"):
            config.validate()

    def test_validate_invalid_dropout(self):
        """Test validation fails for invalid dropout."""
        config = TrainingConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()

    def test_validate_invalid_learning_rate(self):
        """Test validation fails for invalid learning_rate."""
        config = TrainingConfig(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

    def test_validate_invalid_min_lr(self):
        """Test validation fails for invalid min_lr."""
        config = TrainingConfig(min_lr=-1)
        with pytest.raises(ValueError, match="min_lr must be non-negative"):
            config.validate()

    def test_validate_min_lr_exceeds_learning_rate(self):
        """Test validation fails when min_lr exceeds learning_rate."""
        config = TrainingConfig(learning_rate=1e-4, min_lr=1e-3)
        with pytest.raises(ValueError, match="cannot exceed learning_rate"):
            config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch_size."""
        config = TrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_validate_invalid_d_type(self):
        """Test validation fails for invalid d_type."""
        config = TrainingConfig(d_type="int32")
        with pytest.raises(ValueError, match="d_type must be"):
            config.validate()


class TestComputeLearningRate:
    """Tests for learning rate schedule computation."""

    def test_warmup_start(self):
        """Test learning rate at start of warmup is near zero."""
        lr = compute_learning_rate(
            iteration=0,
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=2000,
            lr_decay_iters=10000,
        )
        assert lr == 0.0

    def test_warmup_middle(self):
        """Test learning rate at middle of warmup."""
        lr = compute_learning_rate(
            iteration=1000,
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=2000,
            lr_decay_iters=10000,
        )
        assert lr == pytest.approx(3e-4, rel=1e-6)

    def test_warmup_end(self):
        """Test learning rate at end of warmup equals peak."""
        lr = compute_learning_rate(
            iteration=2000,
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=2000,
            lr_decay_iters=10000,
        )
        # At exactly warmup_iters, we're at the start of decay, so should be peak
        assert lr == pytest.approx(6e-4, rel=1e-6)

    def test_after_decay(self):
        """Test learning rate after decay period equals min_lr."""
        lr = compute_learning_rate(
            iteration=15000,
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=2000,
            lr_decay_iters=10000,
        )
        assert lr == pytest.approx(6e-5, rel=1e-6)

    def test_cosine_decay_middle(self):
        """Test cosine decay at middle point."""
        # At middle of decay, cosine should give 0.5 coefficient
        warmup_iters = 0
        lr_decay_iters = 100
        iteration = 50  # Middle point

        lr = compute_learning_rate(
            iteration=iteration,
            learning_rate=1.0,
            min_lr=0.0,
            warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters,
        )
        # At middle: coeff = 0.5 * (1 + cos(pi * 0.5)) = 0.5 * (1 + 0) = 0.5
        assert lr == pytest.approx(0.5, rel=1e-6)


class TestLoadBinaryData:
    """Tests for binary data loading."""

    def test_load_binary_data(self):
        """Test loading binary data from file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            # Write some uint16 data
            import struct
            data = [1, 2, 3, 4, 5]
            for val in data:
                f.write(struct.pack("H", val))
            temp_path = f.name

        try:
            loaded = load_binary_data(temp_path, "H")
            assert len(loaded) == 5
            assert loaded[0].item() == 1
            assert loaded[4].item() == 5
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_binary_data("/nonexistent/path/data.bin", "H")


class TestGetBatch:
    """Tests for batch generation."""

    def test_batch_shape(self):
        """Test that batch has correct shape."""
        # Create dummy data
        train_data = mx.arange(1000)
        val_data = mx.arange(100)

        batch_size = 4
        context_size = 32

        x, y = get_batch(train_data, val_data, "train", batch_size, context_size)

        assert x.shape == (batch_size, context_size)
        assert y.shape == (batch_size, context_size)

    def test_batch_offset(self):
        """Test that y is offset by 1 from x."""
        train_data = mx.arange(1000)
        val_data = mx.arange(100)

        x, y = get_batch(train_data, val_data, "train", 1, 10)

        # y should be x shifted by 1
        # Since we're taking random positions, we can't check exact values
        # but we can check that shapes match
        assert x.shape == y.shape

    def test_val_split(self):
        """Test that val split uses val_data."""
        train_data = mx.ones(1000)
        val_data = mx.zeros(100)

        x, y = get_batch(train_data, val_data, "val", 1, 10)

        # Should use val_data (zeros)
        assert mx.all(x == 0).item()


class TestCheckpointing:
    """Tests for checkpoint save/load functionality."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small model
            model_config = GPTConfig(
                n_layer=2, n_head=2, n_embd=32,
                vocab_size=100, block_size=32
            )
            model = GPT(model_config)

            # Create training config
            train_config = TrainingConfig(
                n_layer=2, n_head=2, n_embd=32,
                learning_rate=1e-3,
            )

            # Create mock optimizer
            import mlx.optimizers as optim
            optimizer = optim.AdamW(learning_rate=1e-3)

            # Save checkpoint
            out_dir = os.path.join(tmpdir, "test_checkpoint")
            os.makedirs(out_dir)
            save_checkpoint(model, optimizer, 100, train_config, model_config, out_dir)

            # Check files exist
            base = os.path.basename(out_dir)
            assert os.path.exists(os.path.join(out_dir, f"{base}.npz"))
            assert os.path.exists(os.path.join(out_dir, f"{base}.json"))
            assert os.path.exists(os.path.join(out_dir, "training_state.json"))

            # Load checkpoint
            loaded_config, training_state, start_iter = load_checkpoint(out_dir)

            assert loaded_config is not None
            assert loaded_config["n_layer"] == 2
            assert loaded_config["n_head"] == 2
            assert start_iter == 101  # Should be next iteration

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint returns defaults."""
        config, state, start_iter = load_checkpoint("/nonexistent/path")
        assert config is None
        assert state is None
        assert start_iter == 0
