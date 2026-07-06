import torch

from rodiloco.aggregators import Krum, Mean
from rodiloco.attacks import AttackSpec, apply_delta_attack, choose_byzantine, poison_targets


def deltas(n=8, d=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    return [base + 0.05 * torch.randn(d, generator=g) for _ in range(n)]


def test_sign_flip_corrupts_only_byzantine():
    ds = deltas()
    spec = AttackSpec(name="sign_flip", n_byzantine=1, lam=10.0)
    byz = [0]
    out = apply_delta_attack(spec, ds, byz)
    assert torch.allclose(out[0], -10.0 * ds[0])
    assert torch.allclose(out[1], ds[1])  # honest untouched


def test_sign_flip_breaks_mean_but_krum_survives():
    ds = deltas()
    good = Mean()(ds)
    spec = AttackSpec(name="sign_flip", n_byzantine=1, lam=10.0)
    corrupted = apply_delta_attack(spec, ds, [0])
    attacked_mean = Mean()(corrupted)
    krum_out = Krum(n_byzantine=1)(corrupted)
    # the naive mean is dragged far from the honest aggregate; krum stays close
    assert (attacked_mean - good).norm() > (krum_out - good).norm()


def test_scaled_noise_calibrated_to_honest_norm():
    ds = deltas()
    spec = AttackSpec(name="scaled_noise", n_byzantine=1, sigma_mult=1.0)
    out = apply_delta_attack(spec, ds, [0])
    honest_norm = torch.stack([d.norm() for d in ds[1:]]).mean()
    assert torch.allclose(out[0].norm(), honest_norm, rtol=0.05)


def test_targeted_drift_is_marked_inner_and_leaves_deltas_untouched():
    spec = AttackSpec(name="targeted_drift", n_byzantine=1)
    assert spec.is_inner
    ds = deltas()
    out = apply_delta_attack(spec, ds, [0])  # inner attacks don't touch deltas here
    assert all(torch.equal(a, b) for a, b in zip(out, ds))


def test_poison_targets_is_a_permutation():
    y = torch.arange(20) % 16
    py = poison_targets(y, vocab_size=16, seed=3)
    assert py.shape == y.shape
    assert not torch.equal(py, y)  # permutation should move labels


def test_choose_byzantine_count_and_range():
    idx = choose_byzantine(8, 3, seed=0)
    assert len(idx) == 3 and len(set(idx)) == 3
    assert all(0 <= i < 8 for i in idx)
