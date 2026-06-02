"""Pkg-05 spec 04 acceptance: CurriculumScheduler (C5-S1/S2/S3 + C5-L1).

Note (K1): TrainConfig.__post_init__ forbids curriculum fracs of 0.0 / 1.0, so the
oracle_only / infer_only / degenerate edge cases cannot be built via ``replace``.
We test the scheduler's edge logic by setting its computed bounds directly (this is
exactly the state a Pkg-08 ablation subclass that bypasses TrainConfig would reach).
"""
import pytest
import torch
from dataclasses import replace

from hyper_mve.configs import V4Config
from hyper_mve.training.curriculum import CurriculumScheduler


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def scheduler(cfg_medium):
    return CurriculumScheduler(cfg_medium)


def _bounds(s1_end, s2_end, max_steps):
    """A scheduler with directly-set bounds (bypasses TrainConfig 0<s1<s2<1)."""
    sched = CurriculumScheduler(V4Config.from_preset("medium"))
    sched.stage_1_end = s1_end
    sched.stage_2_end = s2_end
    sched.max_steps = max_steps
    return sched


# ====== C5-S1: stage boundaries ======

def test_curriculum_stage_boundaries(scheduler, cfg_medium):
    max_s = cfg_medium.train.max_train_steps
    s1 = int(cfg_medium.train.curriculum_stage_1_end_frac * max_s)
    s2 = int(cfg_medium.train.curriculum_stage_2_end_frac * max_s)

    assert scheduler.stage(0) == "stage_1"
    assert scheduler.stage(s1 - 1) == "stage_1"
    assert scheduler.stage(s1) == "stage_2"
    assert scheduler.stage(s2 - 1) == "stage_2"
    assert scheduler.stage(s2) == "stage_3"
    assert scheduler.stage(max_s + 10_000) == "stage_3"


# ====== C5-S2: Stage 1 full oracle ======

def test_stage_1_full_oracle(scheduler, cfg_medium):
    s1 = int(cfg_medium.train.curriculum_stage_1_end_frac * cfg_medium.train.max_train_steps)
    for step in [0, 1000, s1 // 2, s1 - 1]:
        assert scheduler.oracle_z_mixing_weight(step) == 1.0


# ====== C5-S3: Stage 2 anneal monotone + continuous ======

def test_oracle_mixing_anneal_monotonic(scheduler, cfg_medium):
    max_s = cfg_medium.train.max_train_steps
    s1 = int(cfg_medium.train.curriculum_stage_1_end_frac * max_s)
    s2 = int(cfg_medium.train.curriculum_stage_2_end_frac * max_s)
    steps = list(range(s1, s2, max(1, (s2 - s1) // 10)))
    weights = [scheduler.oracle_z_mixing_weight(s) for s in steps]
    for w0, w1 in zip(weights[:-1], weights[1:]):
        assert w0 > w1
    assert weights[0] == 1.0
    assert weights[-1] < 0.15


def test_oracle_mixing_anneal_continuous(scheduler, cfg_medium):
    max_s = cfg_medium.train.max_train_steps
    s1 = int(cfg_medium.train.curriculum_stage_1_end_frac * max_s)
    s2 = int(cfg_medium.train.curriculum_stage_2_end_frac * max_s)
    for step in range(s1, min(s1 + 100, s2)):
        w_prev = scheduler.oracle_z_mixing_weight(step - 1)
        w_curr = scheduler.oracle_z_mixing_weight(step)
        assert abs(w_curr - w_prev) < 0.01


def test_stage_3_zero_oracle(scheduler, cfg_medium):
    s2 = int(cfg_medium.train.curriculum_stage_2_end_frac * cfg_medium.train.max_train_steps)
    for step in [s2, s2 + 10_000, cfg_medium.train.max_train_steps]:
        assert scheduler.oracle_z_mixing_weight(step) == 0.0


# ====== C5-L1: lambda_b ======

def test_lambda_b_curve_matches_cfg(scheduler, cfg_medium):
    expected = cfg_medium.train.w_belief
    for step in [0, 50_000, 100_000, 150_000, 200_000]:
        assert scheduler.lambda_b(step) == expected


# ====== build_oracle_z_seq pass-through ======

def test_build_oracle_z_seq_passes_through(scheduler):
    types_true = torch.tensor([[[0, 0, 1, 1]] * 5] * 2, dtype=torch.long)  # (2, 5, 4)
    oracle_z = scheduler.build_oracle_z_seq(types_true)
    assert oracle_z.shape == (2, 5, 4, 3, 2)
    assert torch.allclose(oracle_z.sum(dim=-1), torch.ones_like(oracle_z[..., 0]))


# ====== D4: subclass override ======

def test_scheduler_subclass_override():
    class StageBasedLambda(CurriculumScheduler):
        def lambda_b(self, step):
            return {"stage_1": 0.5, "stage_2": 1.0, "stage_3": 1.5}[self.stage(step)]

    cfg = V4Config.from_preset("medium")
    sched = StageBasedLambda(cfg)
    s1 = int(cfg.train.curriculum_stage_1_end_frac * cfg.train.max_train_steps)
    s2 = int(cfg.train.curriculum_stage_2_end_frac * cfg.train.max_train_steps)
    assert sched.lambda_b(0) == 0.5
    assert sched.lambda_b(s1) == 1.0
    assert sched.lambda_b(s2) == 1.5


# ====== invalid boundaries rejected (TrainConfig validates on replace) ======

def test_invalid_stage_boundaries():
    cfg = V4Config.from_preset("medium")
    with pytest.raises((AssertionError, ValueError)):
        cfg_bad = replace(cfg, train=replace(cfg.train,
            curriculum_stage_1_end_frac=0.7,
            curriculum_stage_2_end_frac=0.3,
        ))
        CurriculumScheduler(cfg_bad)


# ====== K1 edge logic (bounds set directly; see module docstring) ======

def test_oracle_only_bounds():
    max_s = 200_000
    sched = _bounds(max_s, max_s, max_s)  # oracle_only
    for step in [0, 50_000, max_s, max_s + 10_000]:
        assert sched.stage(step) == "stage_1"
        assert sched.oracle_z_mixing_weight(step) == 1.0


def test_infer_only_bounds():
    max_s = 200_000
    sched = _bounds(0, 0, max_s)  # infer_only
    for step in [0, 50_000, max_s]:
        assert sched.stage(step) == "stage_3"
        assert sched.oracle_z_mixing_weight(step) == 0.0


def test_degenerate_stage_1_eq_stage_2():
    max_s = 200_000
    sched = _bounds(60_000, 60_000, max_s)
    assert sched.oracle_z_mixing_weight(59_999) == 1.0
    assert sched.oracle_z_mixing_weight(60_000) == 0.0
    assert sched.stage(60_000) == "stage_3"


def test_no_div_by_zero_anneal():
    max_s = 200_000
    for s1, s2 in [(0, 0), (0, max_s // 2), (max_s, max_s),
                   (max_s // 2, max_s), (60_000, 60_000), (0, max_s), (max_s // 2, max_s // 2)]:
        sched = _bounds(s1, s2, max_s)
        for step in [0, 50_000, 100_000, max_s]:
            w = sched.oracle_z_mixing_weight(step)
            assert 0.0 <= w <= 1.0


# ====== state_dict roundtrip ======

def test_state_dict_roundtrip(scheduler):
    state = scheduler.state_dict()
    assert state == {}
    scheduler.load_state_dict(state)  # no-op
