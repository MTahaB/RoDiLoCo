"""Phase 2–4 — the DiLoCo simulation loop.

The central trick that makes the project a solo effort: the ``n`` workers are simulated
**sequentially on one device**. Per outer round, for each worker k:

  1. load the global params θ,
  2. run ``H`` inner AdamW steps on worker k's shard,
  3. store the pseudo-gradient Δ_k = θ_before − θ_after (flat vector).

Then optionally corrupt the byzantine Δ's (Phase 3), aggregate all Δ's through the chosen
``aggregate`` (Phase 2 mean / Phase 4 robust), and take one outer Nesterov step.
Communication is accounted analytically — nothing is physically sent.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .aggregators import build_aggregator
from .attacks import AttackSpec, apply_delta_attack, choose_byzantine, poison_targets
from .data import TokenDataset, make_char_corpus, shard_tokens
from .model import ModelConfig, build_model
from .optim import AdamW, clip_grad_norm_
from .utils import CommMeter, flatten_params, git_hash, load_config, pick_device, set_seed, unflatten_to


class OuterNesterov:
    """SGD with Nesterov momentum over pseudo-gradients — the DiLoCo outer optimizer."""

    def __init__(self, lr: float, momentum: float):
        self.lr = lr
        self.momentum = momentum
        self.velocity: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, theta: list[torch.Tensor], agg_delta: torch.Tensor) -> None:
        if self.velocity is None:
            self.velocity = torch.zeros_like(agg_delta)
        # v <- mu*v + g ;  nesterov update = g + mu*v ;  theta <- theta - lr*update
        self.velocity.mul_(self.momentum).add_(agg_delta)
        update = agg_delta + self.momentum * self.velocity
        flat = flatten_params(theta) - self.lr * update
        for p, new in zip(theta, unflatten_to(flat, theta), strict=True):
            p.copy_(new)


def _inner_train(model, shard_ds: TokenDataset, cfg: dict, rng, poison_seed: int | None) -> None:
    """Run H inner AdamW steps in place on ``model``."""
    opt = AdamW(model.parameters(), lr=cfg["inner_lr"], weight_decay=cfg["weight_decay"])
    model.train()
    for _ in range(cfg["H"]):
        x, y = shard_ds.batch(cfg["batch_size"], rng)
        if poison_seed is not None:
            y = poison_targets(y, model.cfg.vocab_size, poison_seed)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(list(model.parameters()), cfg["grad_clip"])
        opt.step()


def run_diloco(cfg: dict, device: torch.device | None = None) -> dict:
    device = device or pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])

    # data & shards ------------------------------------------------------
    text_path = cfg.get("text_path")
    if text_path and Path(text_path).exists():
        text = Path(text_path).read_text(encoding="utf-8", errors="replace")
    else:
        text = ("the quick brown fox jumps over the lazy dog. " * 800)
    tokens, meta = make_char_corpus(text)
    n_val = max(1, len(tokens) // 10)
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]

    n_workers = cfg["n_workers"]
    shards = shard_tokens(train_tokens, n_workers, iid=cfg.get("iid", True), seed=cfg["seed"])
    seq_len = cfg["seq_len"]
    worker_ds = [TokenDataset(s, seq_len, device) for s in shards]
    val_ds = TokenDataset(val_tokens, seq_len, device)

    # global model + outer optimizer ------------------------------------
    mcfg = ModelConfig(vocab_size=meta["vocab_size"], **cfg["model"])
    global_model = build_model(mcfg).to(device)
    outer = OuterNesterov(cfg["outer_lr"], cfg["outer_momentum"])

    # attack / defense ---------------------------------------------------
    attack = AttackSpec(
        name=cfg.get("attack", "none"),
        n_byzantine=cfg.get("n_byzantine", 0),
        lam=cfg.get("attack_lam", 1.0),
        sigma_mult=cfg.get("attack_sigma_mult", 1.0),
    )
    byz_idx = choose_byzantine(n_workers, attack.n_byzantine, cfg["seed"])
    agg_kwargs = dict(cfg.get("aggregator_kwargs", {}))
    if cfg.get("aggregator") == "krum" and "n_byzantine" not in agg_kwargs:
        agg_kwargs["n_byzantine"] = max(1, attack.n_byzantine)
    aggregate = build_aggregator(cfg.get("aggregator", "mean"), **agg_kwargs)

    comm = CommMeter(param_count=sum(p.numel() for p in global_model.parameters()))
    rng = np.random.default_rng(cfg["seed"])

    history = []
    t0 = time.time()
    for rnd in range(cfg["outer_rounds"]):
        theta = [p.detach().clone() for p in global_model.parameters()]
        theta_flat = flatten_params(theta)

        deltas: list[torch.Tensor] = []
        for k in range(n_workers):
            worker = copy.deepcopy(global_model)
            poison = None
            if attack.is_inner and k in byz_idx:
                poison = cfg["seed"] * 1000 + k  # stable per-worker poisoning permutation
            _inner_train(worker, worker_ds[k], cfg, rng, poison)
            delta_k = theta_flat - flatten_params(list(worker.parameters()))
            deltas.append(delta_k)

        # Phase 3: corrupt byzantine deltas (delta-falsification attacks)
        deltas = apply_delta_attack(attack, deltas, byz_idx)

        # Phase 2/4: aggregate + outer Nesterov step
        agg_delta = aggregate(deltas)
        outer.step(list(global_model.parameters()), agg_delta)
        comm.record_round(n_workers)

        if rnd % cfg["eval_every"] == 0 or rnd == cfg["outer_rounds"] - 1:
            vloss = _evaluate(global_model, val_ds, rng, cfg["batch_size"], cfg.get("eval_batches", 10))
            rec = {
                "round": rnd,
                "val_loss": float(vloss),
                "val_ppl": float(np.exp(min(vloss, 20))),
                "agg_delta_norm": float(agg_delta.norm().item()),
            }
            history.append(rec)
            print(
                f"round {rnd:4d} | val {rec['val_loss']:.4f} | ppl {rec['val_ppl']:.2f} | "
                f"|delta_agg| {rec['agg_delta_norm']:.3f}"
            )

    result = {
        "history": history,
        "final_ppl": history[-1]["val_ppl"] if history else None,
        "git": git_hash(),
        "seed": cfg["seed"],
        "config": {k: cfg[k] for k in cfg if k != "model"},
        "byzantine_workers": byz_idx,
        "comm_total_bytes": comm.total_bytes,
        "comm_reduction_vs_sync": comm.vs_synchronous(cfg["H"]),
        "wall_time_s": round(time.time() - t0, 1),
    }
    if cfg.get("out_dir"):
        out = Path(cfg["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {out}/history.json")
    return result


@torch.no_grad()
def _evaluate(model, ds: TokenDataset, rng, batch_size: int, n_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = ds.batch(batch_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def run_synchronous(cfg: dict, device: torch.device | None = None) -> dict:
    """Phase 2 reference-high curve: fully-synchronous data-parallel training.

    This is the baseline DiLoCo is measured against. It trains one model on the *pooled*
    (unsharded) corpus for the SAME token budget as the DiLoCo run — total inner steps
    = outer_rounds x n_workers x H — but, being synchronous, it "communicates" every step.
    So it defines both the quality ceiling and the high-communication point of the
    "perplexity vs communication" plot.
    """
    device = device or pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])

    text_path = cfg.get("text_path")
    if text_path and Path(text_path).exists():
        text = Path(text_path).read_text(encoding="utf-8", errors="replace")
    else:
        text = ("the quick brown fox jumps over the lazy dog. " * 800)
    tokens, meta = make_char_corpus(text)
    n_val = max(1, len(tokens) // 10)
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]

    seq_len = cfg["seq_len"]
    train_ds = TokenDataset(train_tokens, seq_len, device)
    val_ds = TokenDataset(val_tokens, seq_len, device)

    mcfg = ModelConfig(vocab_size=meta["vocab_size"], **cfg["model"])
    model = build_model(mcfg).to(device)
    opt = AdamW(model.parameters(), lr=cfg["inner_lr"], weight_decay=cfg["weight_decay"])

    total_steps = cfg["outer_rounds"] * cfg["n_workers"] * cfg["H"]
    param_count = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(cfg["seed"])

    history = []
    t0 = time.time()
    eval_every_steps = max(1, total_steps // max(1, cfg["outer_rounds"] // cfg["eval_every"]))
    for step in range(total_steps):
        model.train()
        x, y = train_ds.batch(cfg["batch_size"], rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(list(model.parameters()), cfg["grad_clip"])
        opt.step()
        if step % eval_every_steps == 0 or step == total_steps - 1:
            vloss = _evaluate(model, val_ds, rng, cfg["batch_size"], cfg.get("eval_batches", 10))
            history.append({"step": step, "val_loss": float(vloss), "val_ppl": float(np.exp(min(vloss, 20)))})
            print(f"[sync] step {step:5d}/{total_steps} | val {vloss:.4f} | ppl {np.exp(min(vloss,20)):.2f}")

    result = {
        "mode": "synchronous",
        "history": history,
        "final_ppl": history[-1]["val_ppl"] if history else None,
        "git": git_hash(),
        "seed": cfg["seed"],
        # synchronous DP exchanges one gradient per worker every single step
        "comm_total_bytes": total_steps * cfg["n_workers"] * param_count * 4,
        "total_steps": total_steps,
        "wall_time_s": round(time.time() - t0, 1),
    }
    if cfg.get("out_dir"):
        out = Path(cfg["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {out}/history.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2-4: DiLoCo simulation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--attack")
    ap.add_argument("--n-byzantine", type=int, dest="n_byzantine")
    ap.add_argument("--aggregator")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument(
        "--mode", choices=["diloco", "synchronous"], default="diloco",
        help="'synchronous' runs the reference-high data-parallel baseline",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    for key in ("seed", "attack", "n_byzantine", "aggregator", "out_dir"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.mode == "synchronous":
        run_synchronous(cfg)
    else:
        run_diloco(cfg)


if __name__ == "__main__":
    main()
