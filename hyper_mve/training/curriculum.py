"""CurriculumScheduler — 3-stage curriculum (Pkg-05 spec 04, Ch5.7).

Pure Oracle -> Anneal -> Pure Inference. Lightweight, stateless class: every
method is a pure function of ``global_step`` (the trainer holds the step). This
keeps the scheduler trivially replaceable for Pkg-08 ablation sweeps (D4) and
reusable read-only from Pkg-07 eval.

Stage boundaries (default Medium, max_train_steps from cfg):
    Stage 1 (Pure Oracle):    step <  s1_end = frac1 * max_train_steps
    Stage 2 (Anneal):         s1_end <= step < s2_end = frac2 * max_train_steps
    Stage 3 (Pure Inference): step >= s2_end

The scheduler controls *oracle z injection* (``oracle_z_mixing_weight``) and the
L_belief course weight (``lambda_b``). It is the only object ``compose_total_loss``
talks to for curriculum decisions, so Pkg-08 can subclass it without touching the
loss code.
"""
from __future__ import annotations

import torch

from hyper_mve.configs import V4Config
from hyper_mve.models.belief_losses import build_oracle_z_seq as _build_oracle_z_seq


class CurriculumScheduler:
    """3-stage curriculum scheduler (Ch5.7); see module docstring."""

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.max_steps: int = cfg.train.max_train_steps
        self.stage_1_end: int = int(cfg.train.curriculum_stage_1_end_frac * self.max_steps)
        self.stage_2_end: int = int(cfg.train.curriculum_stage_2_end_frac * self.max_steps)

        # Order constraint only (TrainConfig.__post_init__ already enforces the
        # stricter 0 < s1 < s2 < 1; the <= here additionally tolerates the
        # degenerate oracle_only / infer_only boundaries used by Pkg-08 ablation
        # subclasses that bypass TrainConfig validation).
        assert 0 <= self.stage_1_end <= self.stage_2_end <= self.max_steps, (
            f"Stage boundaries must be monotone: 0 <= {self.stage_1_end} "
            f"<= {self.stage_2_end} <= {self.max_steps}."
        )

    # ------------------------------------------------------------------ stage

    def stage(self, global_step: int) -> str:
        """Return 'stage_1' / 'stage_2' / 'stage_3' for ``global_step``."""
        # Degenerate boundaries first (oracle_only / infer_only ablations).
        if self.stage_1_end == self.max_steps:
            return "stage_1"          # oracle_only: always Stage 1
        if self.stage_2_end == 0:
            return "stage_3"          # infer_only: always Stage 3

        if global_step < self.stage_1_end:
            return "stage_1"
        if global_step < self.stage_2_end:
            return "stage_2"
        return "stage_3"

    # ------------------------------------------------------ oracle injection

    def oracle_z_mixing_weight(self, global_step: int) -> float:
        """Oracle z injection weight in [0, 1].

        Stage 1: 1.0 (full oracle).  Stage 2: linear anneal 1.0 -> 0.0.
        Stage 3: 0.0 (pure BeliefNet inference). The blend is applied by
        ``compose_total_loss``: ``z_main = w * oracle_z + (1 - w) * z_predicted``.
        """
        # Degenerate boundaries (guard against a zero-width anneal interval).
        if self.stage_1_end == self.max_steps:
            return 1.0                # oracle_only
        if self.stage_2_end == 0:
            return 0.0                # infer_only
        if self.stage_1_end == self.stage_2_end:
            # No Stage 2 interval: step the weight 1.0 -> 0.0 at the boundary.
            return 1.0 if global_step < self.stage_1_end else 0.0

        if global_step < self.stage_1_end:
            return 1.0
        if global_step < self.stage_2_end:
            progress = (global_step - self.stage_1_end) / (self.stage_2_end - self.stage_1_end)
            return float(1.0 - progress)
        return 0.0

    # --------------------------------------------------------------- lambda_b

    def lambda_b(self, global_step: int) -> float:
        """L_belief course weight. Default: constant ``cfg.train.w_belief``.

        BeliefNet is trained throughout (Pkg-03 spec 08 §4.2); the curriculum is
        expressed via ``oracle_z_mixing_weight``, not via lambda_b. Pkg-08 may
        subclass and return a stage-dependent curve.
        """
        return float(self.cfg.train.w_belief)

    # --------------------------------------------------- oracle z construction

    def build_oracle_z_seq(self, types_true: torch.Tensor) -> torch.Tensor:
        """Pass-through wrapper of Pkg-03 ``build_oracle_z_seq`` (spec 08 §4.3).

        Args:
            types_true: (B, T, N) int64 (AgentType.value).
        Returns:
            (B, T, N, N-1, 2) one-hot oracle z (agent_id ascending, skip self).
        """
        return _build_oracle_z_seq(types_true)

    # --------------------------------------------------------- serialization

    def state_dict(self) -> dict:
        """No internal state (lightweight); returns an empty dict."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """No-op (no internal state)."""
        return None
