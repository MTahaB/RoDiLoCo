"""Byzantine attacks — Phase 3.

An attack corrupts the pseudo-gradients of the byzantine subset of workers *before*
aggregation. Two families:

* **Δ-falsification** (``sign_flip``, ``scaled_noise``) — the worker trains honestly but
  reports a tampered Δ_k. Cheap, often devastating against a naive mean.
* **inner poisoning** (``targeted_drift``) — the worker trains on corrupted data, so its
  Δ_k has a *plausible norm* but points the wrong way. Stealthier: norm-based defenses
  see nothing anomalous.

``targeted_drift`` is applied inside the training loop (label permutation), so here it is
represented by a marker; the diloco loop reads it and corrupts that worker's targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AttackSpec:
    name: str
    n_byzantine: int = 0  # f: how many of the workers are adversarial
    lam: float = 1.0  # sign-flip scale λ
    sigma_mult: float = 1.0  # scaled-noise σ multiplier (× honest Δ norm)

    @property
    def is_inner(self) -> bool:
        return self.name == "targeted_drift"


def apply_delta_attack(spec: AttackSpec, deltas: list[Tensor], byz_idx: list[int]) -> list[Tensor]:
    """Return a new list of deltas with the byzantine ones corrupted.

    honest_ref norm is estimated from the honest workers so the noise attack is calibrated
    to a realistic magnitude (an attacker who can observe honest scale).
    """
    if spec.n_byzantine == 0 or spec.name in ("none", "targeted_drift"):
        return deltas

    honest = [d for i, d in enumerate(deltas) if i not in byz_idx]
    honest_norm = torch.stack([d.norm() for d in honest]).mean() if honest else deltas[0].norm()

    out = list(deltas)
    for i in byz_idx:
        d = deltas[i]
        if spec.name == "sign_flip":
            out[i] = -spec.lam * d
        elif spec.name == "scaled_noise":
            noise = torch.randn_like(d)
            noise = noise / noise.norm().clamp_min(1e-8) * honest_norm * spec.sigma_mult
            out[i] = noise
        else:
            raise KeyError(f"unknown delta attack '{spec.name}'")
    return out


def poison_targets(targets: Tensor, vocab_size: int, seed: int) -> Tensor:
    """Label permutation for the targeted-drift inner attack.

    A fixed random permutation of the vocabulary is applied to the targets, so the worker
    optimizes a coherent-but-wrong objective (norm stays plausible).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(vocab_size, generator=g).to(targets.device)
    return perm[targets]


def choose_byzantine(n_workers: int, n_byzantine: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_workers, generator=g)[:n_byzantine].tolist()
