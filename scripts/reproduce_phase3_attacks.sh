#!/usr/bin/env bash
# Phase 3 — fragility grid: {3 attacks} x {f in 0,1,2,3} x {H in 125,500}.
# Produces plot #1: "one liar in eight kills pretraining".
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import yaml
from rodiloco.diloco import run_diloco

base = yaml.safe_load(open("configs/attack_signflip.yaml"))
rows = []
for attack in ["sign_flip", "scaled_noise", "targeted_drift"]:
    for f in [0, 1, 2, 3]:
        for H in [125, 500]:
            cfg = dict(base)
            cfg.update(attack=attack, n_byzantine=f, H=H,
                       aggregator="mean",
                       out_dir=f"outputs/p3_{attack}_f{f}_H{H}")
            res = run_diloco(cfg)
            rows.append((attack, f, H, res["final_ppl"]))
            print(f"[{attack} f={f} H={H}] final_ppl={res['final_ppl']:.2f}")
print("\nattack,f,H,final_ppl")
for r in rows:
    print(",".join(map(str, r)))
PY
