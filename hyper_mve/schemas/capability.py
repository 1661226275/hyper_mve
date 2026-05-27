"""CapabilityVector (Ch3.6) — agent heterogeneous capability 4-tuple.

Frozen dataclass with closed-interval range validation in ``__post_init__``
(Decision D7). Sampled once per episode and held constant for that episode.

Field order is fixed to match Ch3.6 documentation: ``(eta, phi_fov, nu, zeta)``.
Renaming or reordering is a breaking change across all downstream packages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ._constants import ETA_RANGE, NU_RANGE, PHI_FOV_RANGE, ZETA_RANGE


@dataclass(frozen=True)
class CapabilityVector:
    """Per-agent capability (Ch3.6), 4 fields.

    Args:
        eta:     harvest speed,    sampled from U[0.5, 1.5]
        phi_fov: field of view,    sampled from U[2.0, 4.0]; env rounds to int
        nu:      move reliability, sampled from U[0.8, 1.0]
        zeta:    carry capacity,   sampled from U[10.0, 30.0]
    """

    eta: float
    phi_fov: float
    nu: float
    zeta: float

    def __post_init__(self) -> None:
        lo, hi = ETA_RANGE
        if not (lo <= self.eta <= hi):
            raise ValueError(f"CapabilityVector.eta={self.eta} ∉ [{lo}, {hi}] (Ch3.6)")
        lo, hi = PHI_FOV_RANGE
        if not (lo <= self.phi_fov <= hi):
            raise ValueError(
                f"CapabilityVector.phi_fov={self.phi_fov} ∉ [{lo}, {hi}] (Ch3.6)"
            )
        lo, hi = NU_RANGE
        if not (lo <= self.nu <= hi):
            raise ValueError(f"CapabilityVector.nu={self.nu} ∉ [{lo}, {hi}] (Ch3.6)")
        lo, hi = ZETA_RANGE
        if not (lo <= self.zeta <= hi):
            raise ValueError(
                f"CapabilityVector.zeta={self.zeta} ∉ [{lo}, {hi}] (Ch3.6)"
            )

    def to_array(self, dtype=np.float32) -> np.ndarray:
        """Pack into a ``(4,)`` numpy array; field order matches Ch3.6."""
        return np.array(
            [self.eta, self.phi_fov, self.nu, self.zeta], dtype=dtype
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CapabilityVector":
        """Inverse of ``to_array``; validates via ``__post_init__``."""
        if arr.shape != (4,):
            raise AssertionError(f"Expected shape (4,), got {arr.shape}")
        return cls(
            eta=float(arr[0]),
            phi_fov=float(arr[1]),
            nu=float(arr[2]),
            zeta=float(arr[3]),
        )

    @property
    def fov_int(self) -> int:
        """Integer Chebyshev radius used by the env's FOV filter."""
        return int(round(self.phi_fov))


def sample_default(rng: np.random.Generator) -> CapabilityVector:
    """Sample one capability from the Ch3.6 default ranges."""
    eta_lo, eta_hi = ETA_RANGE
    fov_lo, fov_hi = PHI_FOV_RANGE
    nu_lo, nu_hi = NU_RANGE
    zeta_lo, zeta_hi = ZETA_RANGE
    return CapabilityVector(
        eta=float(rng.uniform(eta_lo, eta_hi)),
        phi_fov=float(rng.uniform(fov_lo, fov_hi)),
        nu=float(rng.uniform(nu_lo, nu_hi)),
        zeta=float(rng.uniform(zeta_lo, zeta_hi)),
    )


def sample_n(n: int, rng: np.random.Generator) -> tuple[CapabilityVector, ...]:
    """Sample ``n`` independent capabilities."""
    return tuple(sample_default(rng) for _ in range(n))


def to_batch_tensor(
    caps: Sequence[CapabilityVector],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack into ``(N, 4) float32`` tensor for the cap_emb MLP."""
    arr = np.stack([c.to_array() for c in caps])
    out = torch.from_numpy(arr)
    return out.to(device) if device is not None else out
