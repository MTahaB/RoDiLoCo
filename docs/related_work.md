# Related work & the novelty gap

The one-line position: **prior work makes DiLoCo *resilient* (to failures); no one has made
it *robust* (to adversaries).** Fault-tolerance ≠ Byzantine-tolerance.

Verified via literature search, July 2026. RoDiLoCo sits at the empty intersection of two
mature bodies of work.

## The two parent literatures

**A. Low-communication / local-update training (DiLoCo line).** Efficiency under *honest*
workers. None of these study adversaries.
- DiLoCo — inner AdamW ×H, pseudo-gradient, outer Nesterov; ≈ sync quality at ~500× less
  comm on 8 workers. [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)
- Streaming DiLoCo. [arXiv:2501.18512](https://arxiv.org/abs/2501.18512)
- Scaling laws for DiLoCo. [arXiv:2503.09799](https://arxiv.org/abs/2503.09799)
- Understanding Outer Optimizers in Local SGD — formalizes outer LR/momentum/acceleration,
  **no adversaries**. Cite as the paper that defines the regime we then attack.
  [arXiv:2509.10439](https://arxiv.org/html/2509.10439)

**B. Byzantine-robust distributed / federated optimization.** Adversary-aware, but built for
raw **gradients** at **large n** (classification-scale FL), not pseudo-gradients / outer
momentum / n≈8 / LLM pretraining.
- Krum, coordinate-wise trimmed mean, geometric median — the classical robust aggregators
  RoDiLoCo ports and tests for transfer.
- "Mean Aggregator is More Robust than Robust Aggregators under Label Poisoning" — why the
  transfer question is non-trivial. [JMLR 2024](https://arxiv.org/html/2404.13647v2)
- Delayed Momentum Aggregation (DeMoA) — Byz-robust + comm-efficient + momentum, but
  **federated, partial participation, client gradients; explicitly not DiLoCo / outer
  Nesterov over pseudo-gradients / LLM pretraining**. Nearest neighbor in spirit; its
  non-transfer to our regime *is* RQ2. [arXiv:2509.02970](https://arxiv.org/html/2509.02970v1)

## Adjacent but distinct

- **Decoupled DiLoCo — "Resilient Distributed Pre-training"** (DeepMind). Resilience to
  **hardware failures / crash-stop faults** (honest-but-absent workers), *not* Byzantine
  (present-and-lying) workers. The central contrast for our positioning.
  [arXiv:2604.21428](https://arxiv.org/html/2604.21428v1)
- **Byzantine-Robust Decentralized Coordination of LLM Agents** — multi-agent *inference*
  coordination, not training. Name collision only. [arXiv:2507.14928](https://arxiv.org/pdf/2507.14928)

## Honest caveats (state these before a reviewer does)

1. **Trust weighting is not new in the abstract.** Temporal/reputation trust scoring exists
   in FL (e.g. [KeTS 2501.06729](https://arxiv.org/pdf/2501.06729); FedReview;
   ["Byzantines can also Learn from History" 2208.09894](https://arxiv.org/pdf/2208.09894)).
   Our novelty is the *port to the DiLoCo outer regime*, not the invention of trust weighting.
2. **The space is moving fast** (multiple Sept-2025 papers). The gap is real but narrowing —
   which raises the value of moving early and public.

## Net contribution

Byzantine robustness *specifically* in the DiLoCo outer-optimizer regime — a regime that
combines features breaking the assumptions of *both* parents (DiLoCo assumes honesty;
classical robust FL assumes gradients + large n). Three standalone results even if the
proposed method loses: **fragility**, **transfer + tax**, **the regime characterization**.
