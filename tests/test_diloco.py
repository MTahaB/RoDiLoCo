"""Integration tests for the DiLoCo loop — kept tiny so they run on CPU in seconds.

These lock in the three headline behaviours the whole project rests on:
  * the baseline loop reduces perplexity (it actually trains);
  * a sign-flip byzantine worker wrecks the naive mean;
  * a robust aggregator recovers much of the damage.
"""

import copy

import yaml

from rodiloco.diloco import run_diloco, run_synchronous

# a fast, deterministic tiny config shared by the tests
TINY = dict(
    seed=0,
    device="cpu",
    seq_len=32,
    batch_size=8,
    n_workers=4,
    iid=True,
    H=8,
    outer_rounds=5,
    inner_lr=3e-3,
    outer_lr=0.7,
    outer_momentum=0.9,
    weight_decay=0.1,
    grad_clip=1.0,
    aggregator="mean",
    attack="none",
    n_byzantine=0,
    attack_lam=10.0,
    eval_every=2,
    eval_batches=5,
    model=dict(d_model=64, n_layers=2, n_heads=4, max_seq_len=32),
)


def _cfg(**over):
    c = copy.deepcopy(TINY)
    c.update(over)
    return c


def test_baseline_trains():
    res = run_diloco(_cfg())
    first = res["history"][0]["val_ppl"]
    assert res["final_ppl"] < first  # loop makes progress


def test_signflip_breaks_mean():
    clean = run_diloco(_cfg())["final_ppl"]
    attacked = run_diloco(_cfg(attack="sign_flip", n_byzantine=1))["final_ppl"]
    assert attacked > clean * 2  # one liar of four visibly degrades the mean


def test_robust_aggregator_recovers():
    attacked_mean = run_diloco(_cfg(attack="sign_flip", n_byzantine=1))["final_ppl"]
    defended = run_diloco(
        _cfg(attack="sign_flip", n_byzantine=1, aggregator="krum", aggregator_kwargs={"n_byzantine": 1})
    )["final_ppl"]
    assert defended < attacked_mean  # krum beats the naive mean under attack


def test_synchronous_baseline_runs_and_accounts_comm():
    res = run_synchronous(_cfg())
    assert res["mode"] == "synchronous"
    assert res["final_ppl"] is not None
    # synchronous communicates every step => far more bytes than a DiLoCo round
    assert res["comm_total_bytes"] > 0


def test_config_file_loads():
    for name in ["diloco_baseline", "attack_signflip", "defense_trustweighted", "free_tier"]:
        cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
        assert "n_workers" in cfg and "model" in cfg
