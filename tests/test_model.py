"""
Tests for the GPT model implementation.

Run with: pytest tests/test_model.py -v
"""

import pytest
import mlx.core as mx

from model import (
    GPT,
    GPTConfig,
    CausalSelfAttention,
    MLP,
    Block,
    ATTENTION_MASK_VALUE,
    MLP_EXPANSION_FACTOR,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_N_LAYER,
    DEFAULT_N_HEAD,
    DEFAULT_N_EMBD,
)


class TestGPTConfig:
    """Tests for GPTConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GPTConfig()
        assert config.block_size == DEFAULT_BLOCK_SIZE
        assert config.vocab_size == DEFAULT_VOCAB_SIZE
        assert config.n_layer == DEFAULT_N_LAYER
        assert config.n_head == DEFAULT_N_HEAD
        assert config.n_embd == DEFAULT_N_EMBD
        assert config.dropout == 0.0
        assert config.bias is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = GPTConfig(
            block_size=512,
            vocab_size=1000,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.1,
            bias=False,
        )
        assert config.block_size == 512
        assert config.vocab_size == 1000
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
        assert config.dropout == 0.1
        assert config.bias is False

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
        config.validate()  # Should not raise

    def test_validate_invalid_block_size(self):
        """Test validation fails for invalid block_size."""
        config = GPTConfig(block_size=0)
        with pytest.raises(ValueError, match="block_size must be positive"):
            config.validate()

    def test_validate_invalid_vocab_size(self):
        """Test validation fails for invalid vocab_size."""
        config = GPTConfig(vocab_size=-1)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            config.validate()

    def test_validate_invalid_n_embd_n_head(self):
        """Test validation fails when n_embd not divisible by n_head."""
        config = GPTConfig(n_embd=100, n_head=12)
        with pytest.raises(ValueError, match="must be divisible"):
            config.validate()

    def test_validate_invalid_dropout(self):
        """Test validation fails for invalid dropout."""
        config = GPTConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention module."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)

    @pytest.fixture
    def attention(self, config):
        """Create attention layer for testing."""
        return CausalSelfAttention(config)

    def test_init(self, attention, config):
        """Test attention layer initialization."""
        assert attention.n_head == config.n_head
        assert attention.n_embd == config.n_embd

    def test_forward_shape(self, attention, config):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 16
        x = mx.random.normal((batch_size, seq_len, config.n_embd))
        mask = CausalSelfAttention.create_additive_causal_mask(seq_len)

        output, cache = attention(x, mask)

        assert output.shape == (batch_size, seq_len, config.n_embd)
        assert len(cache) == 2  # key and value
        assert cache[0].shape[2] == seq_len  # cached sequence length

    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        N = 10
        mask = CausalSelfAttention.create_additive_causal_mask(N)
        assert mask.shape == (1, 1, N, N)

    def test_causal_mask_is_lower_triangular(self):
        """Test causal mask is lower triangular."""
        N = 5
        mask = CausalSelfAttention.create_additive_causal_mask(N)
        mask_2d = mask.reshape(N, N)

        # Check lower triangular (1s below and on diagonal, 0s above)
        for i in range(N):
            for j in range(N):
                if j <= i:
                    assert mask_2d[i, j].item() == 1.0
                else:
                    assert mask_2d[i, j].item() == 0.0

    def test_invalid_n_embd_n_head(self):
        """Test that invalid n_embd/n_head raises error."""
        config = GPTConfig(n_embd=100, n_head=12, vocab_size=100, block_size=32)
        with pytest.raises(ValueError, match="must be divisible"):
            CausalSelfAttention(config)


class TestMLP:
    """Tests for MLP module."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)

    @pytest.fixture
    def mlp(self, config):
        """Create MLP layer for testing."""
        return MLP(config)

    def test_forward_shape(self, mlp, config):
        """Test MLP forward pass preserves shape."""
        batch_size, seq_len = 2, 16
        x = mx.random.normal((batch_size, seq_len, config.n_embd))

        output = mlp(x)

        assert output.shape == x.shape

    def test_expansion_factor(self, config):
        """Test MLP uses correct expansion factor."""
        mlp = MLP(config)
        # Check intermediate layer size
        expected_intermediate = MLP_EXPANSION_FACTOR * config.n_embd
        assert mlp.c_fc.weight.shape[0] == expected_intermediate


class TestBlock:
    """Tests for transformer Block module."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)

    @pytest.fixture
    def block(self, config):
        """Create Block for testing."""
        return Block(config)

    def test_forward_shape(self, block, config):
        """Test Block forward pass preserves shape."""
        batch_size, seq_len = 2, 16
        x = mx.random.normal((batch_size, seq_len, config.n_embd))
        mask = CausalSelfAttention.create_additive_causal_mask(seq_len)

        output, cache = block(x, mask)

        assert output.shape == x.shape


class TestGPT:
    """Tests for GPT model."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)

    @pytest.fixture
    def model(self, config):
        """Create GPT model for testing."""
        return GPT(config)

    def test_init(self, model, config):
        """Test model initialization."""
        assert model.config == config
        assert len(model.transformer) == config.n_layer

    def test_forward_shape(self, model, config):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 16
        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=mx.int64)

        logits = model(x)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_exceeds_block_size(self, model, config):
        """Test forward raises error when sequence exceeds block_size."""
        seq_len = config.block_size + 10
        x = mx.zeros((1, seq_len), dtype=mx.int64)

        with pytest.raises(ValueError, match="Cannot forward sequence"):
            model(x)

    def test_generate_output_length(self, model, config):
        """Test generate produces correct output length."""
        prompt = mx.array([[1, 2, 3]], dtype=mx.int64)
        max_new_tokens = 5

        output = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)

        assert output.shape[1] == prompt.shape[1] + max_new_tokens

    def test_generate_with_none_input(self, model):
        """Test generate with None input."""
        output = model.generate(None, max_new_tokens=5, temperature=1.0)
        assert output.shape[0] == 1
        assert output.shape[1] == 1 + 5  # 1 initial + 5 generated

    def test_generate_with_top_k(self, model):
        """Test generate with top_k sampling."""
        prompt = mx.array([[1, 2, 3]], dtype=mx.int64)
        output = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
        assert output.shape[1] == prompt.shape[1] + 5

    def test_missing_vocab_size(self):
        """Test that missing vocab_size raises error."""
        config = GPTConfig(vocab_size=None)
        with pytest.raises(ValueError, match="vocab_size must be specified"):
            GPT(config)

    def test_missing_block_size(self):
        """Test that missing block_size raises error."""
        config = GPTConfig(block_size=None, vocab_size=100)
        with pytest.raises(ValueError, match="block_size must be specified"):
            GPT(config)


class TestConstants:
    """Tests for module constants."""

    def test_attention_mask_value(self):
        """Test ATTENTION_MASK_VALUE is a large negative number."""
        assert ATTENTION_MASK_VALUE < -1e8

    def test_mlp_expansion_factor(self):
        """Test MLP_EXPANSION_FACTOR is 4 (standard transformer)."""
        assert MLP_EXPANSION_FACTOR == 4

    def test_default_values(self):
        """Test default configuration values."""
        assert DEFAULT_VOCAB_SIZE == 50304
        assert DEFAULT_BLOCK_SIZE == 1024
        assert DEFAULT_N_LAYER == 12
        assert DEFAULT_N_HEAD == 12
        assert DEFAULT_N_EMBD == 768
