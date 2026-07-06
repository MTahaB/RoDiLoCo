"""Reproducibility, config loading, and param<->vector helpers.

Rule of the project: *a non-reproducible result does not exist*. Every run logs its
seed, its YAML config and the git commit hash.
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Determinism over raw speed for research runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_amp(device: "torch.device", enabled: bool = True):
    """Return ``(use_amp, GradScaler)`` for fp16 mixed precision.

    Enabled only on CUDA (the T4's tensor cores are the whole point). On CPU/MPS the scaler
    is a no-op, so the *same* training-loop code runs everywhere — scale/unscale/step become
    pass-throughs. Halving the forward/backward to fp16 is the biggest single speedup on the
    free T4, on top of the ``_foreach`` optimizer.
    """
    use = bool(enabled) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use)
    return use, scaler


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    """Short commit hash, or 'nogit' outside a repo — stamped into every log."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


# --- pseudo-gradient plumbing -------------------------------------------------
# Aggregators, attacks and defenses all operate on a *flat* Δ vector per worker.
# Keeping a single canonical flatten/unflatten avoids subtle ordering bugs where a
# defense reshapes deltas differently from how the outer optimizer applies them.


@torch.no_grad()
def flatten_params(params: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate a parameter list into one 1-D vector (detached, cloned)."""
    return torch.cat([p.detach().reshape(-1) for p in params])


@torch.no_grad()
def unflatten_to(vec: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a flat vector back into tensors shaped like `like`."""
    out, offset = [], 0
    for p in like:
        n = p.numel()
        out.append(vec[offset : offset + n].reshape(p.shape))
        offset += n
    return out


@dataclass
class CommMeter:
    """Analytic communication accounting (bytes that *would* be exchanged).

    We never physically send anything — workers are simulated sequentially — so
    'communication saved' is computed, not measured. Each outer round exchanges one
    Δ per worker (upload) plus one aggregate broadcast (download).
    """

    param_count: int
    dtype_bytes: int = 4  # fp32 pseudo-gradients
    outer_rounds: int = 0
    workers: int = 0

    def record_round(self, workers: int) -> None:
        self.outer_rounds += 1
        self.workers = workers

    @property
    def bytes_per_round(self) -> int:
        # upload: each worker sends its Δ; download: broadcast aggregate to each worker
        return 2 * self.workers * self.param_count * self.dtype_bytes

    @property
    def total_bytes(self) -> int:
        return self.outer_rounds * self.bytes_per_round

    def vs_synchronous(self, inner_steps: int) -> float:
        """Communication reduction factor vs step-synchronous data-parallel.

        Synchronous DP communicates every inner step; DiLoCo only every H steps.
        """
        return float(inner_steps)  # the headline ~H× (100×+) reduction
