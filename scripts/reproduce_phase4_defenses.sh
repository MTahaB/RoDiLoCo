#!/usr/bin/env bash
# Phase 4 — comparative campaign: {mean, trimmed_mean, krum, geometric_median,
# trust_weighted} x {worst P3 attack} x {f in 0,1,2}.
# Produces plot #2 (ppl vs f per defense) and plot #3 (robustness tax at f=0).
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import yaml
from rodiloco.diloco import run_diloco

base = yaml.safe_load(open("configs/defense_trustweighted.yaml"))
WORST_ATTACK = "sign_flip"   # set to the worst attack found in Phase 3
rows = []
for agg in ["mean", "trimmed_mean", "krum", "geometric_median", "trust_weighted"]:
    for f in [0, 1, 2]:
        cfg = dict(base)
        cfg.update(aggregator=agg, attack=WORST_ATTACK, n_byzantine=f,
                   out_dir=f"outputs/p4_{agg}_f{f}")
        res = run_diloco(cfg)
        rows.append((agg, f, res["final_ppl"]))
        print(f"[{agg} f={f}] final_ppl={res['final_ppl']:.2f}")
print("\naggregator,f,final_ppl")
for r in rows:
    print(",".join(map(str, r)))
PY
