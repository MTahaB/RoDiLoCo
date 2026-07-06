"""A decoder-only Transformer, from scratch.

Deliberately *not* using ``nn.MultiheadAttention``: the point of Phase 1 is to convert
conceptual knowledge of the architecture into an implementation, because you cannot debug
a DiLoCo loop without owning the training loop underneath it.

Components: RMSNorm, RoPE, causal multi-head attention (hand-written), SwiGLU MLP,
pre-norm decoder block, and the stacked model.

Guards against the classic pitfalls:
  * causal mask is applied to the scores *before* softmax (not after);
  * the 1/sqrt(head_dim) scaling is present;
  * RoPE rotates the head_dim axis, in (even, odd) pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None  # defaults to ~8/3 * d_model, rounded, for SwiGLU parity
    max_seq_len: int = 256
    rope_theta: float = 10_000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        # SwiGLU has 3 matrices instead of 2; scale hidden by 2/3 to match param budget,
        # then round to a multiple of 64 for good kernel shapes.
        raw = int(8 / 3 * self.d_model)
        return (raw + 63) // 64 * 64


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        # compute in fp32 for stability, cast back
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    ang = torch.outer(pos, freqs)  # (seq_len, half)
    # duplicate each frequency so it lines up with the (even, odd) interleave below
    ang = torch.cat([ang, ang], dim=-1)  # (seq_len, head_dim)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the last (head_dim) axis. x: (B, H, T, head_dim)."""
    T, d = x.shape[-2], x.shape[-1]
    cos = cos[:T].view(1, 1, T, d)
    sin = sin[:T].view(1, 1, T, d)
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)  # rotate_half
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # scaled_dot_product_attention applies the 1/sqrt(head_dim) scale and, with
        # is_causal=True, the causal mask *inside* the softmax — the correct place.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) * W3 x )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.ff_dim
        self.w1 = nn.Linear(cfg.d_model, h, bias=False)
        self.w3 = nn.Linear(cfg.d_model, h, bias=False)
        self.w2 = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm decoder block: x + attn(norm(x)); x + mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying: fewer params, standard for small LMs
        self.lm_head.weight = self.tok_emb.weight
        self._rope: tuple[Tensor, Tensor] | None = None
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cache(self, T: int, device, dtype) -> tuple[Tensor, Tensor]:
        if self._rope is None or self._rope[0].shape[0] < T or self._rope[0].device != device:
            self._rope = build_rope_cache(
                max(T, self.cfg.max_seq_len), self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        return self._rope

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self._rope_cache(T, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()  # tied with lm_head
        return n


def build_model(cfg: ModelConfig) -> Transformer:
    return Transformer(cfg)


# reference attention used by tests to check equivalence to tolerance 1e-4
def reference_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Naive causal attention, for test cross-checking only. q,k,v: (B,H,T,d)."""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    T = q.shape[-2]
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return attn @ v
