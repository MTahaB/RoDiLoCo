"""Aggregate outputs/*/history.json across seeds into the headline plots + summary table.

Run directories are named  p3_<attack>_f<f>_s<seed>  and  p4_<agg>_f<f>_s<seed>  (the
`_s<seed>` suffix is optional; single-seed runs still work). Results are grouped over seeds
and reported as mean +/- std; a run whose final perplexity hit the exp(20) cap (or is NaN)
counts as *diverged*.

Usage:
    python scripts/plot_results.py fragility 'outputs/p3_*'    results/plot1_fragility.png
    python scripts/plot_results.py defense   'outputs/p4_*'    results/plot2_defense.png
    python scripts/plot_results.py tax       'outputs/p4_*_f0' results/plot3_tax.png
    python scripts/plot_results.py summary   'outputs/p4_*'
"""

from __future__ import annotations

import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CAP = math.exp(20)      # the diverged marker the training loop writes
DIVERGED = 1e6          # anything above this (or None/NaN) is treated as diverged
_NAME = re.compile(r"(p3|p4)_(?P<key>.+)_f(?P<f>\d+)(?:_s(?P<seed>\d+))?$")


def _load(pattern: str):
    for d in sorted(glob.glob(pattern)):
        hp = Path(d) / "history.json"
        if hp.exists():
            yield Path(d).name, json.loads(hp.read_text())


def _val(res) -> tuple[float, bool]:
    v = res.get("final_ppl")
    if v is None or (isinstance(v, float) and math.isnan(v)) or v > DIVERGED:
        return CAP, True
    return float(v), False


def _grouped(pattern: str):
    """key -> {f -> list[(ppl, diverged)]}."""
    g: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for name, res in _load(pattern):
        m = _NAME.match(name)
        if not m:
            continue
        g[m["key"]][int(m["f"])].append(_val(res))
    return g


def _stats(pairs: list[tuple[float, bool]]):
    """(mean, std, n_diverged, n). mean/std use capped values so the cliff shows on plots."""
    vals = np.array([v for v, _ in pairs])
    ndiv = sum(d for _, d in pairs)
    return float(vals.mean()), float(vals.std()), ndiv, len(pairs)


def _line_plot(pattern, dst, title):
    g = _grouped(pattern)
    plt.figure(figsize=(6, 4))
    for key in sorted(g):
        xs = sorted(g[key])
        means = [_stats(g[key][f])[0] for f in xs]
        stds = [_stats(g[key][f])[1] for f in xs]
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=key)
    plt.yscale("log")
    plt.xlabel("byzantine workers f (of 8)")
    plt.ylabel("final validation perplexity (log)")
    plt.title(title)
    plt.legend()
    _save(dst)


def fragility(pattern: str, dst: str) -> None:
    _line_plot(pattern, dst, "Fragility of vanilla DiLoCo (mean aggregation)")


def defense(pattern: str, dst: str) -> None:
    _line_plot(pattern, dst, "Defense transfer under the worst attack")


def tax(pattern: str, dst: str) -> None:
    g = _grouped(pattern)  # only f=0 dirs matched by the caller's glob
    aggs = sorted(g)
    means = [_stats(g[a][0])[0] for a in aggs]
    stds = [_stats(g[a][0])[1] for a in aggs]
    plt.figure(figsize=(6, 4))
    plt.bar(aggs, means, yerr=stds, capsize=4)
    plt.ylabel("final perplexity at f=0 (no attack)")
    plt.title("Robustness tax (lower = cheaper)")
    plt.xticks(rotation=30, ha="right")
    _save(dst)


def summary(pattern: str, dst: str | None = None) -> None:
    """Print a mean +/- std table over seeds (agg/attack x f)."""
    g = _grouped(pattern)
    fs = sorted({f for key in g for f in g[key]})
    header = f"{'run':20s}" + "".join(f"{'f=' + str(f):>16s}" for f in fs)
    print(header)
    print("-" * len(header))
    for key in sorted(g):
        cells = []
        for f in fs:
            if f not in g[key]:
                cell = "-"
            else:
                _, _, ndiv, n = _stats(g[key][f])
                if ndiv == n:
                    cell = "diverged"
                else:
                    good = np.array([v for v, d in g[key][f] if not d])
                    cell = f"{good.mean():.2f}+/-{good.std():.2f}" + (f" ({ndiv}/{n}div)" if ndiv else "")
            cells.append(f"{cell:>16s}")
        print(f"{key:20s}" + "".join(cells))


def comm(pattern: str, dst: str) -> None:
    pts = []
    for name, res in _load(pattern):
        if res.get("final_ppl") and res.get("comm_total_bytes"):
            label = res.get("mode", "diloco")
            m = re.search(r"H(\d+)", name)
            label = f"DiLoCo H={m.group(1)}" if m else ("synchronous (ref)" if label == "synchronous" else label)
            pts.append((res["comm_total_bytes"], res["final_ppl"], label))
    pts.sort()
    plt.figure(figsize=(6, 4))
    plt.scatter([p[0] for p in pts], [p[1] for p in pts])
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
    kind, pattern = sys.argv[1], sys.argv[2]
    dst = sys.argv[3] if len(sys.argv) > 3 else None
    {"fragility": fragility, "defense": defense, "tax": tax,
     "comm": comm, "summary": summary}[kind](pattern, dst)
