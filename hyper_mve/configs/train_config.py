"""TrainConfig — training loop, loss weights and curriculum (Ch5)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass


_VALID_LR_SCHEDULES: tuple[str, ...] = ("warmup_cosine", "cosine", "multistep")


@dataclass(frozen=True)
class TrainConfig:
    """v4 trainer configuration (Pkg-05 consumes this).

    Deprecation note (pkg-08 spec 06 §4 + design D10): the legacy field
    ``use_coord_desc`` was renamed to ``randomize_order``. Both READ
    (``cfg.train.use_coord_desc``) and WRITE
    (``dataclasses.replace(cfg.train, use_coord_desc=False)``) paths still
    work for one-release-minimum, each emitting a ``DeprecationWarning``.
    READ goes through the ``@property`` shim defined below; WRITE is routed
    through the ``_use_coord_desc_compat`` synthetic field +
    ``__init__``-wrap mechanism installed at module import time (see
    ``_install_use_coord_desc_init_wrap`` at the bottom of this file).
    """

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

    # BeliefNet sub-loss weights (v5 Pkg-09: L_regime CE + L_div hinge)
    w_belief_regime: float = 1.0
    w_belief_div: float = 0.01
    belief_div_target_std: float = 0.1
    # DEPRECATED v4 sub-loss weights (unused since the v5 flip; kept only so
    # legacy presets remain constructible until Stage-6 cleanup).
    w_belief_c: float = 1.0
    w_belief_opp: float = 0.5

    # Curriculum boundaries (Ch5.7)
    curriculum_stage_1_end_frac: float = 0.3
    curriculum_stage_2_end_frac: float = 0.7

    # Belief gradient gating (Ch4.6 defence line)
    belief_grad_gating_steps: int = 5000

    # hyper_pred context detach (Pkg-04 spec 01 §3.4, D5): when True, the value
    # path (hyper_pred) receives ctx_aug.detach() so value loss cannot distort
    # the context encoders; reward loss (hyper_rew) still trains them.
    detach_pred_context: bool = True

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
    # [v4-opt 2026-06] z-score noise guard (2agent run diagnosis): rows whose raw
    # per-candidate return std falls below this floor get a UNIFORM pi_mve target.
    # Without it, near-equal candidate returns make the z-score amplify the
    # spa-scenario sampling noise to unit scale and softmax emits a confident but
    # arbitrary target (observed as H_pi_mve drifting up after step ~6000).
    # 0.0 disables the guard (legacy behaviour).
    mve_qstd_floor: float = 0.01

    # 2x2 ablation switches (Ch6.7).
    use_crn: bool = True
    # [pkg-08 spec 06 §4 / design D10] Canonical name (planner-side
    # coordinate-descent agent ordering on / off). Renamed from the legacy
    # ``use_coord_desc`` (still reachable via the @property shim + WRITE-path
    # reconciliation below). Default ``True`` mirrors the historical default
    # byte-identically — no production run, YAML config, or checkpoint
    # changes behaviour after the rename (pkg-08 spec 06 §4.4).
    randomize_order: bool = True
    # [pkg-08 spec 06 §5 / design D10] Abl4 Joint cell, Easy N=2 only.
    # When True, the MVE planner enumerates the joint 6^N action space
    # instead of running coordinate descent (Pkg-05 1-logical-branch downstream
    # patch in mve_planner.py — pkg-08 spec 08 §5).
    mve_joint_enumerate: bool = False

    # Regime-stratified sampling (v5: buckets = episode initial regime g_0;
    # the field name keeps its historical spelling — semantics are per-bucket
    # minimum batch fraction).
    stratified_sampling: bool = True
    stratified_min_per_type_frac: float = 0.3

    # [pkg-08 spec 06 §4.2] Synthetic field — absorbs legacy
    # ``use_coord_desc=…`` kwargs that arrive via the wrapped ``__init__``
    # below. Default is the sentinel ``None`` ("user did not pass
    # ``use_coord_desc=…``"); any non-None value triggers reconciliation
    # in ``__post_init__``. Internal: never read directly by callers (the
    # ``test_use_coord_desc_compat_is_internal`` lock-test asserts this).
    _use_coord_desc_compat: bool | None = None

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

        # [pkg-08 spec 06 §4.2] WRITE-path reconciliation. The
        # ``__init__``-wrap below catches the legacy ``use_coord_desc=…``
        # kwarg and stuffs it into ``_use_coord_desc_compat``. If we see a
        # non-None value here, the user is on the legacy WRITE path; mirror
        # into ``randomize_order`` and emit the deprecation warning with the
        # locked "kwarg is deprecated" wording.
        if self._use_coord_desc_compat is not None:
            compat = bool(self._use_coord_desc_compat)
            if compat != self.randomize_order:
                warnings.warn(
                    "TrainConfig.use_coord_desc kwarg is deprecated "
                    "(pkg-08 spec 06 Lock 2 + design D10); use "
                    "randomize_order instead. The legacy value has been "
                    "mirrored into randomize_order for this config.",
                    DeprecationWarning, stacklevel=3,
                )
                # bypass frozen for the one reconciliation
                object.__setattr__(self, "randomize_order", compat)

    # ------------------------------------------------------------------ alias

    @property
    def use_coord_desc(self) -> bool:
        """DEPRECATED alias for :attr:`randomize_order`.

        Pkg-08 spec 06 §4.1 + Lock 2: ``cfg.train.use_coord_desc`` was
        renamed to ``cfg.train.randomize_order``. Reads through the alias
        emit ``DeprecationWarning``; writes via
        ``dataclasses.replace(cfg.train, use_coord_desc=…)`` are routed
        through the ``__init__``-wrap mechanism installed below.
        """
        warnings.warn(
            "cfg.train.use_coord_desc is deprecated "
            "(pkg-08 spec 06 Lock 2 + design D10); "
            "use cfg.train.randomize_order instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.randomize_order


def _install_use_coord_desc_init_wrap() -> None:
    """Wrap :meth:`TrainConfig.__init__` so the legacy ``use_coord_desc=…``
    kwarg keeps working through ``dataclasses.replace`` and direct
    construction.

    The dataclass-generated ``__init__`` only accepts declared fields. To
    preserve the WRITE-path alias without losing dataclass machinery
    (``__repr__`` / ``__eq__`` / ``__hash__`` / ``replace``), we wrap the
    generated ``__init__`` here: if the caller passed ``use_coord_desc=…``,
    the wrapper rewrites it into ``_use_coord_desc_compat=…`` before
    delegating to the original ``__init__``. Reconciliation then happens in
    ``__post_init__`` (which the dataclass-generated ``__init__`` calls).
    """
    original_init = TrainConfig.__init__

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "use_coord_desc" in kwargs:
            # Rewrite the legacy kwarg into the synthetic-field slot. We
            # always overwrite ``_use_coord_desc_compat`` because
            # ``dataclasses.replace`` round-trips every field including the
            # sentinel (with the original instance's value, which by design
            # is ``None`` outside of a prior reconciliation), so the legacy
            # kwarg is the authoritative user-supplied value.
            kwargs["_use_coord_desc_compat"] = kwargs.pop("use_coord_desc")
        original_init(self, *args, **kwargs)

    __init__.__qualname__ = original_init.__qualname__
    __init__.__name__ = original_init.__name__
    TrainConfig.__init__ = __init__   # type: ignore[method-assign]


_install_use_coord_desc_init_wrap()
