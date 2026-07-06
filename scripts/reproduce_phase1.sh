#!/usr/bin/env bash
# Phase 1 — from-scratch transformer converges on a tiny CPU corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m rodiloco.train --config configs/phase1_smoke.yaml "$@"
