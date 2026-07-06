import torch

from rodiloco.aggregators import (
    GeometricMedian,
    Krum,
    Mean,
    TrimmedMean,
    TrustWeighted,
    build_aggregator,
)


def honest_deltas(n=8, d=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    return [base + 0.05 * torch.randn(d, generator=g) for _ in range(n)]


def test_mean_matches_torch_mean():
    ds = honest_deltas()
    assert torch.allclose(Mean()(ds), torch.stack(ds).mean(0))


def test_trimmed_mean_ignores_outlier():
    ds = honest_deltas()
    clean = TrimmedMean(trim=1)(ds)
    ds[0] = ds[0] + 1000.0  # one wild worker
    trimmed = TrimmedMean(trim=1)(ds)
    # trimmed mean barely moves; naive mean would move a lot
    assert (trimmed - clean).norm() < (Mean()(ds) - clean).norm()


def test_krum_selects_honest_cluster():
    ds = honest_deltas()
    ds[0] = -100 * ds[0]  # byzantine
    out = Krum(n_byzantine=1, multi=1)(ds)
    # krum output should be near an honest delta, far from the poisoned one
    assert (out - ds[1]).norm() < (out - ds[0]).norm()


def test_geometric_median_robust_to_outlier():
    ds = honest_deltas()
    ref = GeometricMedian()(ds)
    ds[0] = ds[0] + 500.0
    poisoned = GeometricMedian()(ds)
    assert (poisoned - ref).norm() < 50.0


def test_trust_weighted_reduces_to_mean_when_consistent():
    tw = TrustWeighted(beta=0.5, tau=0.1)
    ds = honest_deltas(seed=1)
    # warm up several consistent rounds
    for _ in range(6):
        out = tw(ds)
    # with all-consistent workers, weights ~ uniform => close to plain mean
    assert (out - torch.stack(ds).mean(0)).norm() / torch.stack(ds).mean(0).norm() < 0.15


def test_registry_roundtrip():
    for name in ["mean", "trimmed_mean", "krum", "geometric_median", "trust_weighted"]:
        agg = build_aggregator(name)
        out = agg(honest_deltas())
        assert out.shape == (50,)
