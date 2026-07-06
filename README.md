# RoDiLoCo — Robust DiLoCo

**Byzantine-robust low-communication distributed training.**

Low-communication training (DiLoCo and descendants) promises *cross-organisation*,
*over-the-internet* pretraining — yet the entire literature assumes honest workers.
A single corrupted worker at the outer aggregation step can destroy a pretraining run.
RoDiLoCo studies that gap: **how fragile is vanilla DiLoCo, do classical
Byzantine-robust aggregators transfer to the outer-momentum / pseudo-gradient / few-workers
regime, and at what cost ("robustness tax")** — plus a proposed contribution: a
trust-weighted outer optimizer.

The project follows one arc: **reproduce → attack → defend.**

---

## Results

*(headline plots land here once the runs are done — this is the first thing a reader sees.)*

| | |
|---|---|
| **Plot #1** — one liar in eight kills pretraining | `results/plot1_fragility.png` |
| **Plot #2** — defense transfer under attack | `results/plot2_defense.png` |
| **Plot #3** — the robustness tax at f=0 | `results/plot3_tax.png` |

Paper: [`paper/rodiloco.tex`](paper/rodiloco.tex) (arXiv-style writeup).
Related project: **FedAlpha** — Byzantine-robust *federated* optimization; RoDiLoCo ports that
machinery from the inner/gradient regime to the DiLoCo outer/pseudo-gradient regime.

## The central design decision

The whole project fits behind one interface:

```python
def aggregate(deltas: list[Tensor]) -> Tensor: ...
```

Everything is an implementation of `aggregate`:

- **Phase 2** — `mean` (reproduce the DiLoCo baseline)
- **Phase 3** — attacks are wrappers that corrupt a subset of `deltas` *before* aggregation
- **Phase 4** — `trimmed_mean`, `krum`, `geometric_median`, `trust_weighted` (defenses)

The 8 workers are **simulated sequentially on one GPU**: for each worker `k`, load the
global params, run `H` inner AdamW steps on its data shard, store the pseudo-gradient
`Δ_k = θ_before − θ_after`; then aggregate the 8 `Δ`s and take the outer Nesterov step.
No cluster required. Communication is measured *analytically* (bytes that would have been
exchanged), not physically.

---

## Repository layout

```
src/rodiloco/
  tokenizer.py     BPE tokenizer (+ tiktoken GPT-2 fallback)
  model.py         RMSNorm, RoPE, causal MHA (from scratch), SwiGLU, decoder, Transformer
  optim.py         AdamW from scratch, cosine-with-warmup schedule
  data.py          token dataset, i.i.d. + non-i.i.d. sharding
  train.py         single-worker training loop (Phase 1)
  diloco.py        DiLoCo sequential-simulation loop + outer Nesterov (Phase 2)
  aggregators.py   mean / trimmed_mean / krum / geometric_median / trust_weighted
  attacks.py       sign_flip / scaled_noise / targeted_drift (Phase 3)
  generate.py      autoregressive sampling
  utils.py         seeding, config loading, flat<->unflat param vectors, comm accounting
configs/           YAML experiment configs (versioned; a run == a config + a seed)
scripts/           reproduce_*.sh — one entry point per experiment
tests/             shape / equivalence / aggregator / attack unit tests
docs/              related work / novelty positioning, research journal
```

## Quickstart

```bash
uv sync            # or: pip install -e ".[dev]"
pytest             # unit tests (CPU, seconds)
bash scripts/reproduce_phase1.sh   # tiny char-model smoke train on CPU
```

### Run it free (single T4)

The whole project fits on one **free** GPU — workers are simulated sequentially, models are
~10 M params, and metrics are relative degradation, not absolute SOTA. No A100 required.

- `configs/free_tier.yaml` — T4-sized config (10 M-param char model on TinyStories).
- `scripts/prepare_data.py` — download a TinyStories subset to `data/`.
- `notebooks/run_on_kaggle.ipynb` — clone → install → data → P1–P4 → plots on a free Kaggle/Colab T4.

Budget: the full reproduce→attack→defend loop runs inside **~1–2 weeks of Kaggle's 30 GPU-hrs/week**.

See `docs/research_journal.md` for the dated log and `docs/related_work.md` for how this
sits relative to prior DiLoCo and Byzantine-robust work.

## Milestones

| Phase | Deliverable |
|-------|-------------|
| P1 | from-scratch transformer converges |
| P2 | DiLoCo baseline reproduced (ppl ≈ sync at ~100× less comm) |
| P3 | plot #1 — "one liar in eight kills pretraining" |
| P4 | plots #2/#3 — defense transfer + robustness tax + trust-weighted |
| P5 | arXiv-style writeup |

## License

MIT — see [LICENSE](LICENSE).
