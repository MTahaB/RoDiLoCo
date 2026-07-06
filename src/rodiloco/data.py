"""Token data: a contiguous token stream, LM batches, and worker sharding.

Sharding is where the DiLoCo experiment lives:
  * **i.i.d.** (default) — random partition into ``n`` shards. The regime the baseline
    reproduction targets.
  * **non-i.i.d.** — contiguous / label-skewed partition, kept as an ablation knob (the
    plan flags heterogeneity as a later axis).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class TokenDataset:
    """Wraps a flat 1-D array of token ids; yields (x, y) next-token batches."""

    def __init__(self, tokens: np.ndarray | Tensor, seq_len: int, device: str | torch.device = "cpu"):
        if isinstance(tokens, Tensor):
            tokens = tokens.cpu().numpy()
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.seq_len = seq_len
        self.device = device

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len - 1)

    def batch(self, batch_size: int, generator: np.random.Generator) -> tuple[Tensor, Tensor]:
        ix = generator.integers(0, len(self), size=batch_size)
        x = np.stack([self.tokens[i : i + self.seq_len] for i in ix])
        y = np.stack([self.tokens[i + 1 : i + 1 + self.seq_len] for i in ix])
        xt = torch.from_numpy(x).to(self.device)
        yt = torch.from_numpy(y).to(self.device)
        return xt, yt


def shard_tokens(
    tokens: np.ndarray, n_shards: int, *, iid: bool = True, seed: int = 0
) -> list[np.ndarray]:
    """Partition a token stream into ``n_shards`` worker datasets.

    i.i.d.: chunk the stream, then round-robin the chunks so each shard is a random-ish
    mix. non-i.i.d.: hand each worker one contiguous slab (maximally heterogeneous).
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if not iid:
        return list(np.array_split(tokens, n_shards))

    rng = np.random.default_rng(seed)
    # adaptive chunk size: aim for >= ~4 chunks per shard so the round-robin below never
    # leaves a shard empty (happens when the corpus is small, e.g. the smoke test).
    chunk = min(4096, max(1, len(tokens) // (n_shards * 4)))
    n_chunks = len(tokens) // chunk
    chunks = [tokens[i * chunk : (i + 1) * chunk] for i in range(n_chunks)]
    perm = rng.permutation(n_chunks)
    shards: list[list[np.ndarray]] = [[] for _ in range(n_shards)]
    for j, c in enumerate(perm):
        shards[j % n_shards].append(chunks[c])
    return [np.concatenate(s) if s else np.zeros(0, dtype=np.int64) for s in shards]


def make_char_corpus(text: str) -> tuple[np.ndarray, dict]:
    """Tiny byte-level corpus for smoke tests (no external downloads)."""
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    return ids, {"vocab_size": 256}
