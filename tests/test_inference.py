"""
Tests for the inference module.

Run with: pytest tests/test_inference.py -v
"""

import json
import os
import tempfile

import pytest
import mlx.core as mx

from inference import (
    resolve_paths,
    load_model,
    encode_prompt,
    generate_text,
)
from model import GPT, GPTConfig


class TestResolvePaths:
    """Tests for resolve_paths function."""

    def test_resolve_default_paths(self):
        """Test resolving paths from checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directory with expected files
            ckpt_dir = os.path.join(tmpdir, "my_checkpoint")
            os.makedirs(ckpt_dir)

            # Create dummy files
            weights_path = os.path.join(ckpt_dir, "my_checkpoint.npz")
            config_path = os.path.join(ckpt_dir, "my_checkpoint.json")
            open(weights_path, "w").close()
            open(config_path, "w").close()

            # Create mock args
            class Args:
                checkpoint_dir = ckpt_dir
                weights_path = None
                config_path = None

            w, c = resolve_paths(Args())

            assert w == weights_path
            assert c == config_path

    def test_resolve_explicit_paths(self):
        """Test resolving explicit paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            w_path = os.path.join(tmpdir, "custom_weights.npz")
            c_path = os.path.join(tmpdir, "custom_config.json")
            open(w_path, "w").close()
            open(c_path, "w").close()

            class Args:
                checkpoint_dir = tmpdir
                weights_path = w_path
                config_path = c_path

            w, c = resolve_paths(Args())

            assert w == w_path
            assert c == c_path

    def test_missing_weights_file(self):
        """Test error when weights file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "my_checkpoint")
            os.makedirs(ckpt_dir)

            class Args:
                checkpoint_dir = ckpt_dir
                weights_path = None
                config_path = None

            with pytest.raises(FileNotFoundError, match="weights file"):
                resolve_paths(Args())

    def test_missing_config_file(self):
        """Test error when config file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "my_checkpoint")
            os.makedirs(ckpt_dir)

            # Create only weights file
            weights_path = os.path.join(ckpt_dir, "my_checkpoint.npz")
            open(weights_path, "w").close()

            class Args:
                checkpoint_dir = ckpt_dir
                weights_path = None
                config_path = None

            with pytest.raises(FileNotFoundError, match="config file"):
                resolve_paths(Args())


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model(self):
        """Test loading a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small model and save it
            config = GPTConfig(
                n_layer=2, n_head=2, n_embd=32,
                vocab_size=100, block_size=32
            )
            model = GPT(config)

            # Save model
            weights_path = os.path.join(tmpdir, "model.npz")
            config_path = os.path.join(tmpdir, "config.json")

            from mlx.utils import tree_flatten
            flat_params = tree_flatten(model.parameters())
            mx.savez(weights_path, **dict(flat_params))

            with open(config_path, "w") as f:
                json.dump(config.__dict__, f)

            # Load model
            loaded_model = load_model(weights_path, config_path)

            assert loaded_model.config.n_layer == 2
            assert loaded_model.config.n_head == 2
            assert loaded_model.config.n_embd == 32


class TestEncodePrompt:
    """Tests for encode_prompt function."""

    def test_encode_simple_prompt(self):
        """Test encoding a simple prompt."""
        tokenizer, tokens = encode_prompt("Hello, world!")

        assert tokens.shape[0] == 1  # Batch size
        assert tokens.shape[1] > 0  # Has tokens
        assert tokens.dtype == mx.int64

    def test_encode_empty_prompt(self):
        """Test encoding empty prompt uses EOT token."""
        tokenizer, tokens = encode_prompt("")

        assert tokens.shape[0] == 1
        assert tokens.shape[1] == 1  # Just EOT token

    def test_roundtrip_encoding(self):
        """Test that encoding and decoding produces original text."""
        original = "To be, or not to be"
        tokenizer, tokens = encode_prompt(original)

        decoded = tokenizer.decode(tokens[0].tolist())
        assert decoded == original


class TestGenerateText:
    """Tests for generate_text function."""

    def test_generate_text_produces_output(self):
        """Test that generate_text produces output."""
        # Create small model
        config = GPTConfig(
            n_layer=2, n_head=2, n_embd=32,
            vocab_size=50304, block_size=32
        )
        model = GPT(config)
        mx.eval(model.parameters())

        # Encode prompt
        tokenizer, prompt_tokens = encode_prompt("Hello")

        # Generate
        text = generate_text(
            model, tokenizer, prompt_tokens,
            max_new_tokens=5,
            temperature=1.0,
            top_k=None,
        )

        assert isinstance(text, str)
        assert len(text) > 0
        # Output should be longer than input since we generated tokens
        assert len(text) >= len("Hello")

    def test_generate_with_top_k(self):
        """Test generation with top_k sampling."""
        config = GPTConfig(
            n_layer=2, n_head=2, n_embd=32,
            vocab_size=50304, block_size=32
        )
        model = GPT(config)
        mx.eval(model.parameters())

        tokenizer, prompt_tokens = encode_prompt("Test")

        text = generate_text(
            model, tokenizer, prompt_tokens,
            max_new_tokens=5,
            temperature=0.8,
            top_k=10,
        )

        assert isinstance(text, str)
        assert len(text) > 0
