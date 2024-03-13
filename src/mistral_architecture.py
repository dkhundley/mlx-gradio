## Note: This file is more or less a replication of the example directly provided by Apple.
## Please see this line for more info: https://github.com/ml-explore/mlx-examples/blob/main/llms/mistral/mistral.py

# Importing the necessary Python libraries
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor



# Setting the Mistral model arguments
@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 10000



# Creating a class to perform the root mean squared normalization
class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims, ))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims = True) + self.eps)
    
    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output
    


# Creating a class to represent the Attention mechanism
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias = False)
        self.rope = nn.RoPE(args.head_dim, traditional = True, base = args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Preparing the queries, keys, and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        keys = mx.repeat(keys, self.repeats, axis = 1)
        values = mx.repeat(values, self.repeats, axis = 1)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset = key_cache.shape[2])
            keys = self.rope(keys, offset = key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis = 2)
            values = mx.concatenate([value_cache, values], axis = 2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis = -1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)
    


# Creating a class to handle the FeedForward mechanism
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias = False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias = False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
    


# Creating a class to represent the Transformer block
class TransformerBlock(nn.Module):