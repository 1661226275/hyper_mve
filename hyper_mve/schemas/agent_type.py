"""AgentType enum (Ch3.5.1) — α/β preference type encoding.

Decision D1 (see Pkg-01 design.md): use ``IntEnum`` so values can be used
directly as ``nn.Embedding`` indices, packed into ``int8`` numpy arrays for
the buffer, and JSON-serialised as plain integers.
"""
from __future__ import annotations

import enum
from typing import Sequence

import numpy as np
import torch

from ._constants import AGENT_TYPE_ALPHA, AGENT_TYPE_BETA


class AgentType(enum.IntEnum):
    """Agent preference type (Ch3.5.1).

    Members:
        ALPHA (0): purely self-interested, R_i = u_i - ε·1[moved]
        BETA  (1): Fehr-Schmidt inequity-averse,
                   R_i = u_i - ε·1[moved] + φ(c)·ψ(Δ)
    """

    ALPHA = AGENT_TYPE_ALPHA
    BETA = AGENT_TYPE_BETA


def one_hot(t: AgentType, dtype=np.float32) -> np.ndarray:
    """Return a ``(2,)`` one-hot array for the type observation block."""
    arr = np.zeros(2, dtype=dtype)
    arr[int(t)] = 1.0
    return arr


def from_index(idx: int) -> AgentType:
    """``int → AgentType`` reverse lookup; raises ``ValueError`` if out of range."""
    if idx not in (AGENT_TYPE_ALPHA, AGENT_TYPE_BETA):
        raise ValueError(f"AgentType index {idx} ∉ {{0, 1}}")
    return AgentType(idx)


def from_str(name: str) -> AgentType:
    """Parse common spellings ('alpha', 'ALPHA', 'α', 'a', etc.)."""
    norm = name.strip().lower()
    if norm in ("alpha", "α", "a"):
        return AgentType.ALPHA
    if norm in ("beta", "β", "b"):
        return AgentType.BETA
    raise ValueError(f"Cannot parse AgentType from {name!r}")


def count_in_assignment(types: Sequence[AgentType]) -> dict[AgentType, int]:
    """Count members of each type in a type assignment (Ablation 3 helper)."""
    return {
        AgentType.ALPHA: sum(1 for t in types if t == AgentType.ALPHA),
        AgentType.BETA: sum(1 for t in types if t == AgentType.BETA),
    }


def to_long_tensor(
    types: Sequence[AgentType],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Pack ``Sequence[AgentType]`` into an ``(N,) int64`` tensor for nn.Embedding."""
    arr = np.array([int(t) for t in types], dtype=np.int64)
    out = torch.from_numpy(arr)
    return out.to(device) if device is not None else out
