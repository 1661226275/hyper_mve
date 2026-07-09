"""TimeStepRecord — per-step buffer entry (v5, Pkg-09).

7 data fields + ``g`` + ``t`` + ``done``. Stored as numpy arrays so the
buffer stays on CPU (Decision D5); the trainer batches and
``torch.from_numpy``s during sampling.

Key v5 changes vs the v4 record:
- ``g``:     the oracle regime id at this step (int; the L_regime CE target
             and the curriculum oracle-blend source).
- ``row``:   each agent's OWN diagonal-free relationship row ``w_i·``,
             ascending agent_id skipping self (same ordering as the obs
             ``row`` block).
- ``g_hat``: BeliefNet head_regime softmax posterior per agent.
- Removed: ``delta`` / ``tau`` / ``cap`` / ``c_hat`` / ``z_hat`` (the
  Fehr-Schmidt / type / capability / c_t machinery is gone).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TimeStepRecord:
    """Single-step buffer record (v5).

    Shapes assume ``N`` agents, ``A`` discrete actions, observation dim
    ``obs_dim``, and ``|G|`` regimes. The buffer reuses these arrays directly.

    row ordering convention (HARD): ``row[i, k]`` corresponds to
    ``w_{i, j}`` with ``j = (k if k < i else k + 1)`` — ascending agent_id,
    skipping self. It must match the observation ``row`` block and the
    regime family's ``Regime.row(i)``.
    """

    o: np.ndarray              # (N, obs_dim) float32 — joint observation
    a: np.ndarray              # (N,)        int64   — joint action
    r: np.ndarray              # (N,)        float32 — per-agent reward (env-computed)
    pi_mve: np.ndarray         # (N, A)      float32 — MVE planner policy
    v: np.ndarray              # (N,)        float32 — value estimate (NaN allowed)
    row: np.ndarray            # (N, N-1)    float32 — own relationship rows
    g_hat: np.ndarray          # (N, |G|)    float32 — BeliefNet regime posterior
    g: int                     # oracle regime id at this step
    t: int
    done: bool = False

    def __post_init__(self) -> None:
        N = self.o.shape[0]
        per_agent = [self.a, self.r, self.pi_mve, self.v, self.row, self.g_hat]
        if not all(arr.shape[0] == N for arr in per_agent):
            shapes = {name: getattr(self, name).shape for name in
                      ("o", "a", "r", "pi_mve", "v", "row", "g_hat")}
            raise ValueError(
                f"Field N dimension mismatch in TimeStepRecord: {shapes}"
            )
        if self.row.shape[1] != N - 1:
            raise ValueError(
                f"row dim 1 = {self.row.shape[1]}, expected N-1={N - 1}"
            )

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Pack into a numpy-only dict for buffer storage."""
        return {
            "o": self.o,
            "a": self.a,
            "r": self.r,
            "pi_mve": self.pi_mve,
            "v": self.v,
            "row": self.row,
            "g_hat": self.g_hat,
            "g": np.array([self.g], dtype=np.int64),
            "t": np.array([self.t], dtype=np.int32),
            "done": np.array([self.done], dtype=np.bool_),
        }

    @classmethod
    def from_arrays(cls, d: Mapping[str, np.ndarray]) -> "TimeStepRecord":
        """Inverse of ``to_arrays``."""
        return cls(
            o=d["o"], a=d["a"], r=d["r"],
            pi_mve=d["pi_mve"], v=d["v"],
            row=d["row"], g_hat=d["g_hat"],
            g=int(d["g"][0]),
            t=int(d["t"][0]),
            done=bool(d["done"][0]),
        )

    @classmethod
    def empty_belief(
        cls,
        N: int,
        A: int,
        obs_dim: int,
        n_regimes: int,
        t: int = 0,
    ) -> "TimeStepRecord":
        """Placeholder record with neutral belief fields (warmup helper)."""
        return cls(
            o=np.zeros((N, obs_dim), dtype=np.float32),
            a=np.zeros(N, dtype=np.int64),
            r=np.zeros(N, dtype=np.float32),
            pi_mve=np.full((N, A), 1.0 / A, dtype=np.float32),
            v=np.zeros(N, dtype=np.float32),
            row=np.zeros((N, max(N - 1, 1)), dtype=np.float32)[:, : N - 1],
            g_hat=np.full((N, n_regimes), 1.0 / n_regimes, dtype=np.float32),
            g=0,
            t=t,
            done=False,
        )
