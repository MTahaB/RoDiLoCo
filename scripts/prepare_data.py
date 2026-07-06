"""Download a TinyStories subset into a flat text file for the char-level runs.

TinyStories (roneneldan/TinyStories) is small, clean, and converges fast at ~10 M params —
ideal for a free-tier T4. This writes plain UTF-8 text; the loop tokenizes it at the byte
level, so no tokenizer training is needed.

Usage:
    python scripts/prepare_data.py --out data/tinystories_train.txt --max-chars 8_000_000
    python scripts/prepare_data.py --out data/tinystories_train.txt --max-stories 40000

Requires `datasets` (pip install "datasets"). If it is unavailable or offline, the DiLoCo
loop still runs on its built-in fallback corpus (debug scale only).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories_train.txt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-chars", type=int, default=8_000_000, dest="max_chars")
    ap.add_argument("--max-stories", type=int, default=None, dest="max_stories")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"loading roneneldan/TinyStories [{args.split}] (streaming)...")
    ds = load_dataset("roneneldan/TinyStories", split=args.split, streaming=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            f.write(text + "\n\n")
            total += len(text) + 2
            n += 1
            if args.max_stories and n >= args.max_stories:
                break
            if args.max_chars and total >= args.max_chars:
                break
    print(f"wrote {out} — {n} stories, {total:,} chars (~{total // 1_000_000} MB)")


if __name__ == "__main__":
    main()
