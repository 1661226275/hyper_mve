"""Unit tests for ``hyper_mve.utils.schemas.relation`` (v5 Pkg-09).

Hard-gate contracts:

* Exact W tables for every regime in G2 / G4 / g4_ext.
* Relational-reward special cases: ``W = I`` ⇒ selfish; all-ones ⇒ team
  mean; convex-combination bound; asymmetric sign structure.
* Kernel statistics: p=0 never switches (and never consumes RNG);
  p∈{0.5, 1} switch frequencies; holdout restriction.
"""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.utils.configs.env_config import EnvConfig
from hyper_mve.utils.schemas import (
    Regime,
    RegimeFamily,
    build_g2,
    build_g4,
    build_g4_ext,
    compute_relational_rewards,
    get_regime_family,
    sample_initial_regime,
    step_regime,
)


def _env_cfg(N: int = 2, family: str = "g2", **kw) -> EnvConfig:
    base = dict(N=N, L=8, K=8, T_max=100, relation_family=family)
    base.update(kw)
    return EnvConfig(**base)


# ---------------------------------------------------------------------------
# Exact W tables
# ---------------------------------------------------------------------------

def test_g2_exact_w_values():
    fam = build_g2(lam=1.0)
    assert fam.size == 5
    assert fam.N == 2
    assert fam.names() == (
        "mutual_coop", "mutual_comp", "asym_exploit", "asym_exploited", "neutral",
    )
    expected = {
        "mutual_coop": [[1.0, 1.0], [1.0, 1.0]],
        "mutual_comp": [[1.0, -1.0], [-1.0, 1.0]],
        "asym_exploit": [[1.0, -1.0], [1.0, 1.0]],     # agent 0 hostile, 1 supportive
        "asym_exploited": [[1.0, 1.0], [-1.0, 1.0]],   # mirror
        "neutral": [[1.0, 0.0], [0.0, 1.0]],
    }
    for reg in fam.regimes:
        np.testing.assert_array_equal(reg.w_array(), np.array(expected[reg.name], np.float32))


def test_g2_intensity_scales_off_diagonal_only():
    fam = build_g2(lam=0.5)
    W = fam.regimes[0].w_array()  # mutual_coop
    np.testing.assert_array_equal(W, np.array([[1.0, 0.5], [0.5, 1.0]], np.float32))


def test_g4_block_structure():
    fam = build_g4(lam=1.0)
    assert fam.size == 5 and fam.N == 4
    all_coop = fam.regimes[0].w_array()
    assert (all_coop == 1.0).all()
    all_comp = fam.regimes[1].w_array()
    assert (np.diagonal(all_comp) == 1.0).all()
    off = all_comp[~np.eye(4, dtype=bool)]
    assert (off == -1.0).all()
    pair = fam.regimes[2].w_array()  # pair_01_23
    assert pair[0, 1] == 1.0 and pair[2, 3] == 1.0
    assert pair[0, 2] == -1.0 and pair[1, 3] == -1.0


def test_g4_ext_trios():
    fam = build_g4_ext(lam=1.0)
    assert fam.size == 9
    trio = fam.regimes[5].w_array()  # trio_012_3
    assert trio[0, 1] == 1.0 and trio[1, 2] == 1.0
    assert trio[0, 3] == -1.0 and trio[3, 0] == -1.0 and trio[3, 3] == 1.0


def test_regime_row_excludes_diagonal():
    fam = build_g2(lam=1.0)
    asym = fam.regimes[2]  # asym_exploit: W = [[1,-1],[1,1]]
    np.testing.assert_array_equal(asym.row(0), np.array([-1.0], np.float32))
    np.testing.assert_array_equal(asym.row(1), np.array([1.0], np.float32))
    rows = fam.rows_stack()
    assert rows.shape == (5, 2, 1)


def test_regime_validation():
    with pytest.raises(ValueError, match="diagonal"):
        Regime(0, "bad_diag", ((0.5, 1.0), (1.0, 1.0)))
    with pytest.raises(ValueError, match="∉"):
        Regime(0, "oob", ((1.0, 2.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="contiguous"):
        RegimeFamily("f", 2, (Regime(1, "x", ((1.0, 0.0), (0.0, 1.0))),))


# ---------------------------------------------------------------------------
# get_regime_family (config resolution + caching)
# ---------------------------------------------------------------------------

def test_get_regime_family_from_cfg():
    fam = get_regime_family(_env_cfg(N=2, family="g2"))
    assert fam.name == "g2" and fam.size == 5
    # cached: identical object for identical (family, λ)
    assert get_regime_family(_env_cfg(N=2, family="g2")) is fam


def test_get_regime_family_n_mismatch():
    with pytest.raises(ValueError, match="requires N=2"):
        get_regime_family(_env_cfg(N=4, family="g2"))


def test_env_config_validates_relation_fields():
    with pytest.raises(ValueError, match="relation_family"):
        _env_cfg(family="g3")
    with pytest.raises(ValueError, match="regime_switch_prob"):
        _env_cfg(regime_switch_prob=1.5)
    with pytest.raises(ValueError, match="sum to 1"):
        _env_cfg(regime_prior=(0.5, 0.5, 0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="alpha"):
        _env_cfg(alpha=0.0)
    with pytest.raises(ValueError, match="train_regime_ids"):
        _env_cfg(train_regime_ids=())
    # defaults stay valid for a legacy-style config (regimes never consumed)
    _env_cfg(N=4, family="g4")


# ---------------------------------------------------------------------------
# Sampling kernel
# ---------------------------------------------------------------------------

def test_sample_initial_regime_uniform_frequencies():
    fam = build_g2()
    rng = np.random.default_rng(0)
    draws = np.array([sample_initial_regime(fam, rng) for _ in range(10_000)])
    freq = np.bincount(draws, minlength=5) / draws.size
    assert (np.abs(freq - 0.2) < 0.02).all(), freq


def test_sample_initial_regime_allowed_and_prior():
    fam = build_g2()
    rng = np.random.default_rng(1)
    draws = {sample_initial_regime(fam, rng, allowed_ids=(0, 1, 4)) for _ in range(200)}
    assert draws <= {0, 1, 4}
    # prior renormalized over allowed subset
    prior = (0.0, 0.0, 0.0, 0.0, 1.0)
    assert sample_initial_regime(fam, rng, prior=prior) == 4
    with pytest.raises(ValueError, match="zero"):
        sample_initial_regime(fam, rng, prior=prior, allowed_ids=(0, 1))
    with pytest.raises(ValueError, match="length"):
        sample_initial_regime(fam, rng, prior=(0.5, 0.5))


def test_step_regime_p0_is_identity_and_rng_free():
    fam = build_g2()
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    for g in range(5):
        assert step_regime(g, fam, rng_a, switch_prob=0.0) == g
    # p=0 consumed no randomness: streams still aligned
    assert rng_a.random() == rng_b.random()


def test_step_regime_p1_always_switches():
    fam = build_g2()
    rng = np.random.default_rng(3)
    for _ in range(100):
        assert step_regime(2, fam, rng, switch_prob=1.0) != 2


def test_step_regime_switch_frequency():
    fam = build_g2()
    rng = np.random.default_rng(11)
    switches = sum(step_regime(0, fam, rng, switch_prob=0.5) != 0 for _ in range(10_000))
    assert abs(switches / 10_000 - 0.5) < 0.02


def test_step_regime_respects_allowed_and_single_regime():
    fam = build_g2()
    rng = np.random.default_rng(5)
    for _ in range(50):
        nxt = step_regime(0, fam, rng, switch_prob=1.0, allowed_ids=(0, 1))
        assert nxt == 1
    # only one allowed regime -> must stay
    assert step_regime(0, fam, rng, switch_prob=1.0, allowed_ids=(0,)) == 0


# ---------------------------------------------------------------------------
# Relational reward
# ---------------------------------------------------------------------------

def test_reward_identity_w_is_selfish():
    u = np.array([3.0, 0.5, 2.0], np.float32)
    moved = np.array([True, False, True])
    r = compute_relational_rewards(
        harvests=u, moved_mask=moved, W=np.eye(3), epsilon_move=0.01,
    )
    np.testing.assert_allclose(r, u - 0.01 * moved.astype(np.float32), rtol=1e-6)


def test_reward_all_ones_is_team_mean():
    u = np.array([4.0, 0.0], np.float32)
    W = build_g2(1.0).regimes[0].w_array()  # mutual_coop, all ones
    r = compute_relational_rewards(
        harvests=u, moved_mask=np.zeros(2, bool), W=W, epsilon_move=0.01,
    )
    np.testing.assert_allclose(r, [2.0, 2.0], rtol=1e-6)


def test_reward_mutual_comp_zero_sum_pair():
    u = np.array([4.0, 1.0], np.float32)
    W = build_g2(1.0).regimes[1].w_array()  # [[1,-1],[-1,1]]
    r = compute_relational_rewards(
        harvests=u, moved_mask=np.zeros(2, bool), W=W, epsilon_move=0.01,
    )
    np.testing.assert_allclose(r, [1.5, -1.5], rtol=1e-6)


def test_reward_asymmetric_signs():
    # asym_exploit: agent 0 hostile (w01=-1), agent 1 supportive (w10=+1)
    u = np.array([4.0, 2.0], np.float32)
    W = build_g2(1.0).regimes[2].w_array()
    r = compute_relational_rewards(
        harvests=u, moved_mask=np.zeros(2, bool), W=W, epsilon_move=0.01,
    )
    np.testing.assert_allclose(r, [1.0, 3.0], rtol=1e-6)
    # supportive agent gains from the other's harvest; hostile agent is
    # penalized by it
    assert r[1] > u[1] / 2 and r[0] < u[0]


def test_reward_convex_combination_bound():
    rng = np.random.default_rng(42)
    for fam in (build_g2(), build_g4(), build_g4_ext(), build_g2(0.3)):
        for reg in fam.regimes:
            u = rng.uniform(0.0, 10.0, size=fam.N).astype(np.float32)
            moved = rng.random(fam.N) < 0.5
            r = compute_relational_rewards(
                harvests=u, moved_mask=moved, W=reg.w_array(), epsilon_move=0.01,
            )
            assert (np.abs(r + 0.01 * moved.astype(np.float32)) <= u.max() + 1e-5).all()


def test_reward_rejects_bad_diagonal():
    with pytest.raises(AssertionError, match="diagonal"):
        compute_relational_rewards(
            harvests=np.ones(2, np.float32),
            moved_mask=np.zeros(2, bool),
            W=np.array([[0.5, 1.0], [1.0, 1.0]]),
            epsilon_move=0.01,
        )
