"""Phase 1 — the single-worker training loop.

The reference loop DiLoCo sits on top of. Produces the P1 deliverable: a clean loss curve
and English-looking samples. Runs on CPU at toy scale for the smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import TokenDataset, make_char_corpus
from .generate import generate
from .model import ModelConfig, build_model
from .optim import AdamW, clip_grad_norm_, cosine_warmup_lr
from .utils import git_hash, load_config, make_amp, pick_device, set_seed


def train_single(cfg: dict, device: torch.device | None = None) -> dict:
    device = device or pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])

    # data ---------------------------------------------------------------
    text_path = cfg.get("text_path")
    if text_path and Path(text_path).exists():
        text = Path(text_path).read_text(encoding="utf-8", errors="replace")
    else:
        # self-contained smoke corpus
        text = ("the quick brown fox jumps over the lazy dog. " * 400)
    tokens, meta = make_char_corpus(text)
    n_val = max(1, len(tokens) // 10)
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]

    seq_len = cfg["seq_len"]
    train_ds = TokenDataset(train_tokens, seq_len, device)
    val_ds = TokenDataset(val_tokens, seq_len, device)

    # model --------------------------------------------------------------
    mcfg = ModelConfig(vocab_size=meta["vocab_size"], **cfg["model"])
    model = build_model(mcfg).to(device)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_amp, scaler = make_amp(device, cfg.get("amp", True))

    rng = np.random.default_rng(cfg["seed"])
    steps = cfg["steps"]
    history = []
    t0 = time.time()
    for step in range(steps):
        lr = cosine_warmup_lr(
            step, base_lr=cfg["lr"], warmup=cfg["warmup"], total=steps, min_lr=cfg["lr"] * 0.1
        )
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        x, y = train_ds.batch(cfg["batch_size"], rng)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # unscale before clipping so the norm is in real units
        gnorm = clip_grad_norm_(list(model.parameters()), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()

        if step % cfg["eval_every"] == 0 or step == steps - 1:
            vloss = evaluate(model, val_ds, rng, cfg["batch_size"], cfg.get("eval_batches", 10))
            rec = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(vloss),
                "val_ppl": float(np.exp(min(vloss, 20))),
                "lr": lr,
                "grad_norm": float(gnorm),
            }
            history.append(rec)
            print(
                f"step {step:5d} | train {rec['train_loss']:.4f} | "
                f"val {rec['val_loss']:.4f} | ppl {rec['val_ppl']:.2f} | lr {lr:.2e}"
            )

    result = {
        "history": history,
        "git": git_hash(),
        "seed": cfg["seed"],
        "num_params": model.num_params(),
        "wall_time_s": round(time.time() - t0, 1),
    }

    # a small sample, so the log shows it "looks like English"
    ctx = torch.tensor([[train_tokens[0]]], dtype=torch.long, device=device)
    out = generate(model, ctx, 120, temperature=0.8, top_k=40)[0].tolist()
    result["sample"] = bytes(int(t) % 256 for t in out).decode("utf-8", errors="replace")
    print("sample:", result["sample"])

    if cfg.get("out_dir"):
        _dump(cfg, result, model)
    return result


@torch.no_grad()
def evaluate(model, ds: TokenDataset, rng, batch_size: int, n_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = ds.batch(batch_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def _dump(cfg: dict, result: dict, model) -> None:
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), out / "model.pt")
    print(f"wrote {out}/history.json and model.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: single-worker transformer training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.steps is not None:
        cfg["steps"] = args.steps
    train_single(cfg)


if __name__ == "__main__":
    main()
