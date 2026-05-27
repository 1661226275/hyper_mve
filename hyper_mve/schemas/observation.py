"""Six-block observation layout (Ch3.7).

Static dimension constants and padding helpers. All dimensions are computed
without instantiating any env, so models can size their input layers from
``ObservationLayout.total_dim(N, K)`` alone.

**Self-Info principle (Ch3.7)**: the ``type`` block contains only the
*agent's own* type one-hot (2-dim). Opponent types are inferred by the
BeliefNet (Pkg-03).

The ``neighbor`` block adds a ``presence_flag`` (extra 1 dim per slot)
relative to the bare Ch3.7 table so the model can distinguish "FOV-occluded
slot" from "visible neighbor that just chose NOOP".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ObservationBlockSpec:
    """Static specification for a single observation block (documentation only)."""

    name: str
    per_item_dim: int
    item_count_formula: str       # e.g. "1", "K", "N-1"


class ObservationLayout:
    """Six-block observation layout (Ch3.7).

    Block order (fixed, not reorderable):
        1. ``self``       — agent's own state, 4 dims
        2. ``resource``   — FOV-visible resource cells, K × 3 (padded)
        3. ``neighbor``   — FOV-visible other agents, (N-1) × 9 (padded, includes presence_flag)
        4. ``global``     — c_t + episode_time_remaining_ratio, 2 dims
        5. ``capability`` — own CapabilityVector, 4 dims
        6. ``type``       — own type one-hot (Self-Info: 2 dims, not 2N)
    """

    SELF_DIM: int = 4
    RESOURCE_PER_ITEM: int = 3          # (rel_dx, rel_dy, q_norm)
    NEIGHBOR_PER_ITEM: int = 9          # (rel_dx, rel_dy, action_onehot[6], presence_flag)
    GLOBAL_DIM: int = 2                 # (c_t, time_remaining_ratio)
    CAPABILITY_DIM: int = 4             # CapabilityVector flattened
    TYPE_DIM: int = 2                   # own one-hot only (Self-Info)
    NEIGHBOR_ACTION_DIM: int = 6        # one-hot length for last_action

    BLOCK_ORDER: tuple[str, ...] = (
        "self", "resource", "neighbor", "global", "capability", "type",
    )

    @staticmethod
    def block_dim(block_name: str, N: int, K: int) -> int:
        """Flattened dimension of a single block."""
        if block_name == "self":
            return ObservationLayout.SELF_DIM
        if block_name == "resource":
            return K * ObservationLayout.RESOURCE_PER_ITEM
        if block_name == "neighbor":
            return (N - 1) * ObservationLayout.NEIGHBOR_PER_ITEM
        if block_name == "global":
            return ObservationLayout.GLOBAL_DIM
        if block_name == "capability":
            return ObservationLayout.CAPABILITY_DIM
        if block_name == "type":
            return ObservationLayout.TYPE_DIM
        raise ValueError(f"Unknown block name: {block_name}")

    @staticmethod
    def total_dim(N: int, K: int) -> int:
        """Total per-agent observation dim = sum of all 6 blocks.

        Formula: 4 + 3K + 9(N-1) + 2 + 4 + 2 = 12 + 3K + 9(N-1).

        Reference values:
            Easy   (N=2, K=8):  4 + 24 + 9   + 2 + 4 + 2 = 45
            Medium (N=4, K=20): 4 + 60 + 27  + 2 + 4 + 2 = 99
            Hard   (N=8, K=40): 4 + 120 + 63 + 2 + 4 + 2 = 195
        """
        return sum(
            ObservationLayout.block_dim(b, N, K)
            for b in ObservationLayout.BLOCK_ORDER
        )

    @staticmethod
    def block_offset(block_name: str, N: int, K: int) -> tuple[int, int]:
        """``(start, end)`` indices of a block in the flattened observation."""
        if block_name not in ObservationLayout.BLOCK_ORDER:
            raise ValueError(f"Unknown block: {block_name}")
        start = 0
        for b in ObservationLayout.BLOCK_ORDER:
            d = ObservationLayout.block_dim(b, N, K)
            if b == block_name:
                return (start, start + d)
            start += d
        raise RuntimeError("unreachable")


def pad_resource_block(
    visible: Iterable[tuple[float, float, float]],
    K: int,
) -> np.ndarray:
    """Pad a list of visible resource cells ``(rel_dx, rel_dy, q_norm)`` to ``K``.

    Returns a ``(K * 3,)`` flattened float32 array with unused slots zeroed.
    Input longer than ``K`` is truncated.
    """
    out = np.zeros((K, 3), dtype=np.float32)
    for i, item in enumerate(visible):
        if i >= K:
            break
        out[i] = item
    return out.reshape(-1)


def pad_neighbor_block(
    visible: Iterable[tuple[float, float, np.ndarray, bool]],
    N: int,
) -> np.ndarray:
    """Pad visible neighbors to ``N-1`` slots with a presence flag.

    Each ``visible`` entry is ``(rel_dx, rel_dy, last_action_onehot[6], present)``.
    Returns ``((N-1) * 9,)`` flattened float32 array.
    """
    slots = N - 1
    out = np.zeros((slots, ObservationLayout.NEIGHBOR_PER_ITEM), dtype=np.float32)
    for i, (dx, dy, act_oh, presence) in enumerate(visible):
        if i >= slots:
            break
        out[i, 0] = dx
        out[i, 1] = dy
        out[i, 2:8] = np.asarray(act_oh, dtype=np.float32)
        out[i, 8] = 1.0 if presence else 0.0
    return out.reshape(-1)


def slice_block(obs: np.ndarray, block_name: str, N: int, K: int) -> np.ndarray:
    """Slice a single block out of a flattened observation (debug / viz)."""
    start, end = ObservationLayout.block_offset(block_name, N, K)
    return obs[start:end]
