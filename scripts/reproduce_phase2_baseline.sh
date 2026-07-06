#!/usr/bin/env bash
# Phase 2 — DiLoCo baseline (mean aggregation, no attack) + synchronous reference.
# Produces the "perplexity vs communication" plot: the sync baseline is the quality
# ceiling / high-comm point; DiLoCo H in {50,125,500} are the low-comm points.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) synchronous reference-high baseline
python - <<'PY'
import yaml
from rodiloco.diloco import run_synchronous
cfg = yaml.safe_load(open("configs/diloco_baseline.yaml"))
cfg["out_dir"] = "outputs/sync_baseline"
run_synchronous(cfg)
PY

# 2) DiLoCo H sweep
for H in 50 125 500; do
  python - "$H" <<'PY'
import sys, yaml
from rodiloco.diloco import run_diloco
cfg = yaml.safe_load(open("configs/diloco_baseline.yaml"))
cfg["H"] = int(sys.argv[1])
cfg["out_dir"] = f"outputs/diloco_baseline_H{sys.argv[1]}"
run_diloco(cfg)
PY
done

# 3) the plot (glob handled inside python; matches sync_baseline + diloco_baseline_H*)
python scripts/plot_results.py comm 'outputs/*baseline*' results/plotP2_comm.png
