"""Turn outputs/*/history.json into the three headline plots.

Usage:
    python scripts/plot_results.py fragility   outputs/p3_*        results/plot1_fragility.png
    python scripts/plot_results.py defense      outputs/p4_*        results/plot2_defense.png
    python scripts/plot_results.py tax          outputs/p4_*_f0     results/plot3_tax.png
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(pattern: str) -> list[tuple[str, dict]]:
    out = []
    for d in sorted(glob.glob(pattern)):
        hp = Path(d) / "history.json"
        if hp.exists():
            out.append((Path(d).name, json.loads(hp.read_text())))
    return out


def fragility(pattern: str, dst: str) -> None:
    # group by attack, x=f, y=final_ppl
    series = defaultdict(list)
    for name, res in load(pattern):
        m = re.search(r"p3_(\w+?)_f(\d+)", name)
        if not m:
            continue
        attack, f = m.group(1), int(m.group(2))
        series[attack].append((f, res["final_ppl"]))
    plt.figure(figsize=(6, 4))
    for attack, pts in series.items():
        pts.sort()
        xs, ys = zip(*pts, strict=True)
        plt.plot(xs, ys, marker="o", label=attack)
    plt.xlabel("byzantine workers f (of 8)")
    plt.ylabel("final validation perplexity")
    plt.title("Fragility of vanilla DiLoCo (mean aggregation)")
    plt.legend()
    _save(dst)


def defense(pattern: str, dst: str) -> None:
    series = defaultdict(list)
    for name, res in load(pattern):
        m = re.search(r"p4_(\w+?)_f(\d+)", name)
        if not m:
            continue
        agg, f = m.group(1), int(m.group(2))
        series[agg].append((f, res["final_ppl"]))
    plt.figure(figsize=(6, 4))
    for agg, pts in series.items():
        pts.sort()
        xs, ys = zip(*pts, strict=True)
        plt.plot(xs, ys, marker="o", label=agg)
    plt.xlabel("byzantine workers f (of 8)")
    plt.ylabel("final validation perplexity")
    plt.title("Defense transfer under the worst attack")
    plt.legend()
    _save(dst)


def tax(pattern: str, dst: str) -> None:
    # robustness tax: final_ppl at f=0 per aggregator vs the mean baseline
    bars = {}
    for name, res in load(pattern):
        m = re.search(r"p4_(\w+?)_f0", name)
        if m:
            bars[m.group(1)] = res["final_ppl"]
    plt.figure(figsize=(6, 4))
    names = list(bars)
    plt.bar(names, [bars[n] for n in names])
    plt.ylabel("final perplexity at f=0 (no attack)")
    plt.title("Robustness tax (lower = cheaper)")
    plt.xticks(rotation=30, ha="right")
    _save(dst)


def comm(pattern: str, dst: str) -> None:
    """Plot #P2: final perplexity vs total communication (bytes, log scale).

    Expects the synchronous baseline dir plus one dir per DiLoCo H value; each glob match
    must contain a history.json with 'final_ppl' and 'comm_total_bytes'.
    """
    pts = []
    for name, res in load(pattern):
        if res.get("final_ppl") and res.get("comm_total_bytes"):
            label = res.get("mode", "diloco")
            m = re.search(r"H(\d+)", name)
            if m:
                label = f"DiLoCo H={m.group(1)}"
            elif label == "synchronous":
                label = "synchronous (ref)"
            pts.append((res["comm_total_bytes"], res["final_ppl"], label))
    pts.sort()
    plt.figure(figsize=(6, 4))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.scatter(xs, ys)
    for x, y, lab in pts:
        plt.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("total communication (bytes, log)")
    plt.ylabel("final validation perplexity")
    plt.title("DiLoCo: quality vs communication")
    _save(dst)


def _save(dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(dst, dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    kind, pattern, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    {"fragility": fragility, "defense": defense, "tax": tax, "comm": comm}[kind](pattern, dst)
