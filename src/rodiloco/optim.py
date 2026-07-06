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

        # Same AdamW math as the readable per-parameter version, but batched with the
        # ``torch._foreach_*`` ops so the whole parameter list updates in a handful of kernel
        # launches instead of ~O(#tensors) tiny ones. On GPU this is the difference between
        # being launch-overhead-bound and compute-bound (~5-8x fewer launches per step).
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            params, grads, ms, vs = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                ms.append(state["m"])
                vs.append(state["v"])
            if not params:
                continue

            # all params in a group step together => a single shared step counter
            group["t"] = group.get("t", 0) + 1
            t = group["t"]

            # m <- beta1*m + (1-beta1)*g ;  v <- beta2*v + (1-beta2)*g^2
            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, grads, alpha=1 - beta1)
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1 - beta2)

            bias1 = 1 - beta1**t
            bias2 = 1 - beta2**t
            # denom = sqrt(v)/sqrt(bias2) + eps
            denom = torch._foreach_sqrt(vs)
            torch._foreach_div_(denom, math.sqrt(bias2))
            torch._foreach_add_(denom, eps)

            if wd != 0:  # decoupled weight decay (the "W" in AdamW)
                torch._foreach_mul_(params, 1 - lr * wd)
            torch._foreach_addcdiv_(params, ms, denom, value=-lr / bias1)
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
    """Global L2 gradient clipping; returns the pre-clip total norm.

    Branchless and batched: we compute one clip coefficient ``min(1, max_norm/total)`` and
    always multiply by it (multiplying by 1.0 is a no-op). Avoiding the Python ``if clip < 1``
    keeps a GPU scalar off the CPU, so there is no device sync per step — important inside the
    hot inner loop.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = torch._foreach_norm(grads, 2)
    total = torch.norm(torch.stack(list(norms)), 2)
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, coef)
    return total
