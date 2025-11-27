"""
Pytest configuration and shared fixtures for tests.
"""

import os
import sys
import tempfile
import warnings

import pytest
import mlx.core as mx

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def small_config():
    """Create a small GPTConfig for fast tests."""
    from model import GPTConfig
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=100,
        block_size=32,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def small_model(small_config):
    """Create a small GPT model for fast tests."""
    from model import GPT
    model = GPT(small_config)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_tokens():
    """Create sample token data for testing."""
    return mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int64)


@pytest.fixture(autouse=True)
def suppress_configurator_warning():
    """Suppress configurator deprecation warning in all tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="configurator")
        yield


@pytest.fixture
def mock_training_data(temp_dir):
    """Create mock training data files."""
    import struct

    # Create train.bin
    train_path = os.path.join(temp_dir, "train.bin")
    with open(train_path, "wb") as f:
        for i in range(1000):
            f.write(struct.pack("H", i % 100))

    # Create val.bin
    val_path = os.path.join(temp_dir, "val.bin")
    with open(val_path, "wb") as f:
        for i in range(100):
            f.write(struct.pack("H", i % 100))

    return temp_dir
