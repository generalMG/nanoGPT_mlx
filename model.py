"""
GPT model implementation using MLX for Apple Silicon.

This module provides a minimal, educational implementation of a GPT-style
transformer model optimized for Apple Silicon using the MLX framework.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Constants for numerical stability and model configuration
ATTENTION_MASK_VALUE = -1e9  # Large negative value for masked attention positions
MLP_EXPANSION_FACTOR = 4  # Standard transformer MLP expansion ratio
DEFAULT_VOCAB_SIZE = 50304  # GPT-2 vocab (50257) rounded up for efficiency
DEFAULT_BLOCK_SIZE = 1024  # Default context window size
DEFAULT_N_LAYER = 12  # Default number of transformer layers
DEFAULT_N_HEAD = 12  # Default number of attention heads
DEFAULT_N_EMBD = 768  # Default embedding dimension


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.

    Implements scaled dot-product attention with causal masking to prevent
    attending to future tokens. Supports KV-caching for efficient inference.

    Args:
        config: GPTConfig instance containing model hyperparameters.

    Attributes:
        c_attn: Combined linear projection for query, key, and value.
        c_proj: Output projection layer.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
    """

    def __init__(self, config: "GPTConfig") -> None:
        """Initialize the attention layer with the given configuration."""
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"Embedding dimension ({config.n_embd}) must be divisible by "
                f"number of heads ({config.n_head})"
            )
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            mask: Causal mask tensor.
            cache: Optional tuple of (key_cache, value_cache) for incremental decoding.

        Returns:
            Tuple of (output tensor, (key, value) cache for next step).
        """
        B, T, C = x.shape

        query, key, value = mx.split(self.c_attn(x), 3, axis=2)
        key = key.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        query = query.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        value = value.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            key = mx.concatenate([key_cache, key], axis=2)
            value = mx.concatenate([value_cache, value], axis=2)

        att = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(key.shape[3]))
        mask = mask.reshape(1, 1, T, T)
        att = mx.where(mask[:, :, :T, :T] == 0, att, ATTENTION_MASK_VALUE)

        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y, (key, value)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        """
        Create a causal attention mask.

        Args:
            N: Sequence length.
            dtype: Data type for the mask tensor.

        Returns:
            Lower triangular mask of shape (1, 1, N, N).
        """
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)
    

class MLP(nn.Module):
    """
    Feed-forward network (MLP) block for transformer.

    Implements a two-layer MLP with GELU activation and dropout.
    Uses the standard transformer expansion factor of 4x.

    Args:
        config: GPTConfig instance containing model hyperparameters.
    """

    def __init__(self, config: "GPTConfig") -> None:
        """Initialize the MLP with the given configuration."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_EXPANSION_FACTOR * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(MLP_EXPANSION_FACTOR * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of the same shape as input.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    """
    Single transformer block with pre-LayerNorm architecture.

    Combines causal self-attention and MLP with residual connections.
    Uses pre-normalization (LayerNorm before each sub-layer) for
    improved training stability.

    Args:
        config: GPTConfig instance containing model hyperparameters.
    """

    def __init__(self, config: "GPTConfig") -> None:
        """Initialize the transformer block with the given configuration."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, affine=True, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, affine=True, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            mask: Causal attention mask.
            cache: Optional KV cache for incremental decoding.

        Returns:
            Tuple of (output tensor, updated KV cache).
        """
        att, cache = self.attn(self.ln_1(x), mask, cache)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, cache


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model hyperparameters.

    Attributes:
        block_size: Maximum sequence length (context window size).
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads per layer.
        n_embd: Embedding dimension.
        dropout: Dropout probability (0.0 = no dropout).
        bias: Whether to use bias in linear layers and LayerNorm.
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    vocab_size: int = DEFAULT_VOCAB_SIZE
    n_layer: int = DEFAULT_N_LAYER
    n_head: int = DEFAULT_N_HEAD
    n_embd: int = DEFAULT_N_EMBD
    dropout: float = 0.0
    bias: bool = True

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
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


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) language model.

    A decoder-only transformer architecture for autoregressive language modeling.
    Implements token and position embeddings, a stack of transformer blocks,
    and a final projection to vocabulary logits.

    Args:
        config: GPTConfig instance containing model hyperparameters.

    Attributes:
        config: The model configuration.
        wte: Token embedding layer.
        wpe: Positional embedding layer.
        transformer: List of transformer blocks.
        ln_f: Final layer normalization.
        out_proj: Output projection to vocabulary.
    """

    def __init__(self, config: GPTConfig) -> None:
        """
        Initialize the GPT model.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If vocab_size or block_size is not set in config.
        """
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("vocab_size must be specified in config")
        if config.block_size is None:
            raise ValueError("block_size must be specified in config")
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.transformer = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, affine=True, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def _sample_next_token(self, x: mx.array, temperature: float) -> mx.array:
        """
        Sample the next token from hidden states (used for cached generation).

        Args:
            x: Hidden states tensor.
            temperature: Sampling temperature.

        Returns:
            Sampled token indices.
        """
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1 / temperature))
        return y

    def generate(
        self,
        idx: mx.array,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Input token indices of shape (batch_size, seq_len).
                 If None, starts from a zero token.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. Higher values increase diversity.
            top_k: If set, only sample from the top-k most likely tokens.

        Returns:
            Token indices including both input and generated tokens.
        """
        if idx is None:
            idx = mx.zeros((1, 1), dtype=mx.int64)
        batch_size = idx.shape[0]

        for _ in range(max_new_tokens):
            # Truncate to block_size if necessary
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                sorted_logits = mx.sort(logits, axis=-1)
                kth_indices = mx.full(
                    (batch_size, 1),
                    sorted_logits.shape[-1] - k,
                    dtype=mx.int32,
                )
                kth_values = mx.take_along_axis(sorted_logits, kth_indices, axis=-1)
                logits = mx.where(logits < kth_values, ATTENTION_MASK_VALUE, logits)

            idx_next = mx.random.categorical(logits, axis=-1)
            idx_next = mx.expand_dims(idx_next.astype(mx.int64), axis=1)

            idx = mx.concatenate([idx, idx_next], axis=1)

        return idx
    
    def _forward_transformer(
        self,
        x: mx.array,
        pos: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        build_cache: bool = False
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass through the transformer layers.

        Args:
            x: Input token indices.
            pos: Position indices.
            mask: Optional causal attention mask.
            cache: Optional list of KV caches for each layer.
            build_cache: Whether to build and return KV cache.

        Returns:
            Tuple of (hidden states, KV cache or None).
        """
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.transformer[i](x, mask=None, cache=cache[i])
        else:
            for block in self.transformer:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        return x, kv_cache if build_cache else cache

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass to compute logits for next token prediction.

        Args:
            x: Input token indices of shape (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).

        Raises:
            ValueError: If sequence length exceeds block_size.
        """
        b, t = x.shape
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.config.block_size}"
            )
        pos = mx.arange(0, t, 1, dtype=x.dtype)
        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])

        x, _ = self._forward_transformer(x, pos, mask=mask)
        return self.out_proj(x)
