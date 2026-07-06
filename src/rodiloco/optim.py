"""AdamW from scratch and a cosine-with-warmup LR schedule.

This AdamW is reused verbatim as the DiLoCo *inner* optimizer in Phase 2, so it lives in
its own module. Implemented against the decoupled-weight-decay formulation (Loshchilov &
Hutter): weight decay is applied to the parameters directly, not folded into the gradient.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
                step_size = lr / bias1

                # decoupled weight decay (the "W" in AdamW)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)
        return loss


def cosine_warmup_lr(step: int, *, base_lr: float, warmup: int, total: int, min_lr: float = 0.0) -> float:
    """Linear warmup then cosine decay to ``min_lr``.

    Skipping warmup is one of the listed instabilities — this schedule makes it explicit.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def clip_grad_norm_(params, max_norm: float) -> Tensor:
    """Global L2 gradient clipping; returns the pre-clip total norm."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
    clip = max_norm / (total + 1e-6)
    if clip < 1:
        for g in grads:
            g.mul_(clip)
    return total
