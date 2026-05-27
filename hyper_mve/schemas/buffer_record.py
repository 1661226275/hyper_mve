"""TimeStepRecord — per-step buffer entry (Ch5.6.1, v4 revision Pass 2).

10 data fields + ``t`` + ``done``. Stored as numpy arrays so the buffer stays
on CPU (Decision D5); the trainer batches and ``torch.from_numpy``s during
sampling.

Key v4 changes vs v4.7 buffer record:
- ``delta``: instantaneous unfairness Δ_i^(t) = u_i - mean_{j≠i} u_j
- ``tau``:   AgentType packed as int8
- ``cap``:   per-agent CapabilityVector flattened
- ``c_hat``: scalar (N,) — Ch4.2.3 Head 1 sigmoid output, **not** (N, d_c)
- ``z_hat``: (N, N-1, 2) — Ch4.2.3 Head 2, agent_id ascending skipping self
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TimeStepRecord:
    """Single-step buffer record (Ch5.6.1).

    Shapes assume ``N`` agents, ``A`` discrete actions, observation dim
    ``obs_dim``. The buffer reuses these arrays directly.

    z_hat ordering convention (HARD, Oracle-supervised L_opp):
        For agent ``i``, ``z_hat[i, k]`` corresponds to
        ``opponent_id = (k if k < i else k + 1)``  — i.e. ascending agent_id,
        skipping self. Mis-ordering silently corrupts the L_opp CE loss
        (Ch4.5.2).
    """

    o: np.ndarray              # (N, obs_dim) float32 — joint observation
    a: np.ndarray              # (N,)        int64   — joint action
    r: np.ndarray              # (N,)        float32 — per-agent reward (env-computed)
    delta: np.ndarray          # (N,)        float32 — instantaneous unfairness
    pi_mve: np.ndarray         # (N, A)      float32 — MVE planner policy
    v: np.ndarray              # (N,)        float32 — value estimate (NaN allowed)
    tau: np.ndarray            # (N,)        int8    — AgentType.value
    cap: np.ndarray            # (N, 4)      float32 — CapabilityVector flatten
    c_hat: np.ndarray          # (N,)        float32 — BeliefNet head_c scalar
    z_hat: np.ndarray          # (N, N-1, 2) float32 — BeliefNet head_opp softmax
    t: int
    done: bool = False

    def __post_init__(self) -> None:
        N = self.o.shape[0]
        per_agent = [self.a, self.r, self.delta, self.pi_mve, self.v,
                     self.tau, self.cap, self.c_hat, self.z_hat]
        if not all(arr.shape[0] == N for arr in per_agent):
            shapes = {name: getattr(self, name).shape for name in
                      ("o", "a", "r", "delta", "pi_mve", "v",
                       "tau", "cap", "c_hat", "z_hat")}
            raise ValueError(
                f"Field N dimension mismatch in TimeStepRecord: {shapes}"
            )
        if self.z_hat.shape[1] != N - 1:
            raise ValueError(
                f"z_hat dim 1 = {self.z_hat.shape[1]}, expected N-1={N - 1}"
            )

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Pack into a numpy-only dict for buffer storage."""
        return {
            "o": self.o,
            "a": self.a,
            "r": self.r,
            "delta": self.delta,
            "pi_mve": self.pi_mve,
            "v": self.v,
            "tau": self.tau,
            "cap": self.cap,
            "c_hat": self.c_hat,
            "z_hat": self.z_hat,
            "t": np.array([self.t], dtype=np.int32),
            "done": np.array([self.done], dtype=np.bool_),
        }

    @classmethod
    def from_arrays(cls, d: Mapping[str, np.ndarray]) -> "TimeStepRecord":
        """Inverse of ``to_arrays``."""
        return cls(
            o=d["o"], a=d["a"], r=d["r"], delta=d["delta"],
            pi_mve=d["pi_mve"], v=d["v"], tau=d["tau"],
            cap=d["cap"], c_hat=d["c_hat"], z_hat=d["z_hat"],
            t=int(d["t"][0]),
            done=bool(d["done"][0]),
        )

    @classmethod
    def empty_belief(
        cls,
        N: int,
        A: int,
        obs_dim: int,
        t: int = 0,
    ) -> "TimeStepRecord":
        """Placeholder record with neutral belief fields (Stage 1 warmup helper)."""
        return cls(
            o=np.zeros((N, obs_dim), dtype=np.float32),
            a=np.zeros(N, dtype=np.int64),
            r=np.zeros(N, dtype=np.float32),
            delta=np.zeros(N, dtype=np.float32),
            pi_mve=np.full((N, A), 1.0 / A, dtype=np.float32),
            v=np.zeros(N, dtype=np.float32),
            tau=np.zeros(N, dtype=np.int8),
            cap=np.zeros((N, 4), dtype=np.float32),
            c_hat=np.full(N, 0.5, dtype=np.float32),
            z_hat=np.full((N, N - 1, 2), 0.5, dtype=np.float32),
            t=t,
            done=False,
        )
