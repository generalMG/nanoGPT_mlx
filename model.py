import math

import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass

import pdb

class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = False):
        super().__init__()
        self.use_bias = bias
        if affine:
            if bias:
                self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims
    
    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}, bias={self.use_bias}"
    
    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        out = (self.weight * x) if "weight" in self else x
        return out + self.bias if self.use_bias else out
        

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def __call__(self, x, mask, cache=None):
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
        att = mx.where(mask[:, :, :T, :T] == 0, att, float('-1e9'))
        
        att = mx.softmax(att.astype(mx.float32), axis=-1).astype(att.dtype)
        att = self.attn_dropout(att)
        y = (att @ value).transpose(0, 2, 1, 3).reshape(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y, (key, value)
    
    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        return mx.tril(mx.ones([N, N])).reshape(1, 1, N, N).astype(dtype)
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def __call__(self, x, mask, cache=None):
        att, cache = self.attn(self.ln_1(x), mask, cache)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, cache


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.transformer = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def _sample_next_token(self, x, temperature):
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1 / temperature))
        return y
    
    def generate(self, idx: mx.array, max_new_tokens=256, temperature=1.0, top_k=None):
        if idx is None:
            idx = mx.zeros((1, 1), dtype=mx.int64)
        batch_size = idx.shape[0]
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                sorted_logits = mx.sort(logits, axis=-1)
                kth_indices = mx.full(
                    (batch_size, 1),
                    sorted_logits.shape[-1] - k,
                    dtype=mx.int32,
                )
                kth_values = mx.take_along_axis(sorted_logits, kth_indices, axis=-1)
                logits = mx.where(logits < kth_values, float("-1e9"), logits)

            idx_next = mx.random.categorical(logits, axis=-1)
            idx_next = mx.expand_dims(idx_next.astype(mx.int64), axis=1)

            idx = mx.concatenate([idx, idx_next], axis=1)

        return idx
    
    def _forward_transformer(self, x: mx.array, pos: mx.array, mask=None, cache=None, build_cache=False):
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
    
    def __call__(self, x):
        b, t = x.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=x.dtype)
        mask = CausalSelfAttention.create_additive_causal_mask(x.shape[1])

        x, _ = self._forward_transformer(x, pos, mask=mask)
        return self.out_proj(x)
