"""Autoregressive sampling — for the Phase-1 'looks like English' sanity check."""

from __future__ import annotations

import torch
from torch import Tensor

from .model import Transformer


@torch.no_grad()
def generate(
    model: Transformer,
    idx: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Tensor:
    model.eval()
    max_len = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
