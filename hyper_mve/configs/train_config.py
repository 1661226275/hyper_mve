"""TrainConfig — training loop, loss weights and curriculum (Ch5)."""
from __future__ import annotations

from dataclasses import dataclass


_VALID_LR_SCHEDULES: tuple[str, ...] = ("warmup_cosine", "cosine", "multistep")


@dataclass(frozen=True)
class TrainConfig:
    """v4 trainer configuration (Pkg-05 consumes this)."""

    # Total training budget (presets may override; default is Medium baseline)
    max_train_steps: int = 1_000_000

    # Batch + buffer
    batch_size: int = 256
    buffer_size: int = 5000
    min_buffer_size: int = 1000

    # Collection / training cadence
    episodes_per_iter: int = 8
    train_steps_per_iter: int = 8

    # MuZero unroll
    unroll_K: int = 5
    n_step: int = 5
    gamma: float = 0.95

    # Optimiser
    lr: float = 1e-4
    lr_min: float = 5e-6
    adam_eps: float = 1e-5
    grad_clip: float = 10.0

    # LR schedule
    lr_schedule: str = "warmup_cosine"
    lr_warmup_steps: int = 5000

    # Loss weights (Ch5.6.3)
    w_policy: float = 1.0
    w_value: float = 0.25
    w_reward: float = 3.0
    w_consist: float = 0.5
    w_belief: float = 1.0

    # BeliefNet sub-loss weights (Ch4.5)
    w_belief_c: float = 1.0
    w_belief_opp: float = 0.5
    w_belief_div: float = 0.01
    belief_div_target_std: float = 0.1

    # Curriculum boundaries (Ch5.7)
    curriculum_stage_1_end_frac: float = 0.3
    curriculum_stage_2_end_frac: float = 0.7

    # Belief gradient gating (Ch4.6 defence line)
    belief_grad_gating_steps: int = 5000

    # EMA target net (v4.4)
    ema_tau: float = 0.99

    # Exploration
    epsilon_init: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 28_000

    # MVE planner (Pkg-05)
    mve_samples: int = 50
    mve_depth: int = 5
    mve_temperature: float = 1.0

    # 2x2 ablation switches (Ch6.7)
    use_crn: bool = True
    use_coord_desc: bool = True

    # Type-stratified sampling (Ch5.6.5)
    stratified_sampling: bool = True
    stratified_min_per_type_frac: float = 0.3

    def __post_init__(self) -> None:
        s1 = self.curriculum_stage_1_end_frac
        s2 = self.curriculum_stage_2_end_frac
        if not (0.0 < s1 < s2 < 1.0):
            raise ValueError(
                f"Curriculum stage boundaries must satisfy 0 < s1({s1}) < s2({s2}) < 1"
            )
        if self.lr_schedule not in _VALID_LR_SCHEDULES:
            raise ValueError(
                f"Unknown lr_schedule: {self.lr_schedule!r} (valid: {_VALID_LR_SCHEDULES})"
            )
        if not (0.0 < self.lr_min <= self.lr):
            raise ValueError(
                f"lr_min({self.lr_min}) must be in (0, lr={self.lr}]"
            )
        if self.batch_size < 1 or self.buffer_size < 1:
            raise ValueError("batch_size and buffer_size must be positive")
