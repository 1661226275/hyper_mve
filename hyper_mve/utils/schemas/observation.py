"""Five-block v5 observation layout (Pkg-09, ``envs.relation_commons``).

Static dimension constants and padding helpers. All dimensions are computed
without instantiating any env, so models can size their input layers from
``RelationObservationLayout.total_dim(N, K)`` alone.

**Self-Info principle**: the ``row`` block contains only the *agent's own*
relationship row ``w_i·``. Others' rows / the regime id are never observed —
they are the BeliefNet's inference target.

The v4 six-block ``ObservationLayout`` (FOV masking, c_t global slot,
capability + type blocks) was deleted with resource_commons in the Stage-6
cleanup; the name survives as an alias to the v5 layout.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class RelationObservationLayout:
    """Five-block v5 observation layout (Pkg-09, ``envs.relation_commons``).

    Differences from the retired v4 layout:

    * no FOV — ``resource`` holds **all** K cells in index order (also makes
      the block permutation-stable) and ``neighbor`` always sees all others
      (``presence_flag`` kept ``= 1`` for slot-layout stability);
    * ``global`` shrinks to 1 dim (``time_remaining_ratio``; c_t removed);
    * ``capability`` + ``type`` blocks are replaced by ``row`` — the agent's
      **own** relationship row ``w_i·`` (N-1 dims, ascending ``j`` skipping
      self, same ordering as the ``neighbor`` block). Self-Info discipline:
      others' rows are never observed, only inferred (BeliefNet).

    Block order (fixed):
        1. ``self``     — 4 dims (cum-harvest normalized by ``T_max·η``, η=1)
        2. ``resource`` — K × 3 ``(rel_dx, rel_dy, q_norm)``
        3. ``neighbor`` — (N-1) × 9 ``(rel_dx, rel_dy, action_onehot[6], presence)``
        4. ``global``   — 1 dim ``(time_remaining_ratio,)``
        5. ``row``      — (N-1) dims, own ``w_i·``
    """

    SELF_DIM: int = 4
    RESOURCE_PER_ITEM: int = 3
    NEIGHBOR_PER_ITEM: int = 9
    GLOBAL_DIM: int = 1                 # (time_remaining_ratio,)
    NEIGHBOR_ACTION_DIM: int = 6

    BLOCK_ORDER: tuple[str, ...] = (
        "self", "resource", "neighbor", "global", "row",
    )

    @staticmethod
    def block_dim(block_name: str, N: int, K: int) -> int:
        """Flattened dimension of a single block."""
        if block_name == "self":
            return RelationObservationLayout.SELF_DIM
        if block_name == "resource":
            return K * RelationObservationLayout.RESOURCE_PER_ITEM
        if block_name == "neighbor":
            return (N - 1) * RelationObservationLayout.NEIGHBOR_PER_ITEM
        if block_name == "global":
            return RelationObservationLayout.GLOBAL_DIM
        if block_name == "row":
            return N - 1
        raise ValueError(f"Unknown block name: {block_name}")

    @staticmethod
    def total_dim(N: int, K: int) -> int:
        """Total per-agent observation dim = 5 + 3K + 10(N-1).

        Reference values:
            rel_duo  (N=2, K=8):  4 + 24 + 9  + 1 + 1 = 39
            rel_quad (N=4, K=20): 4 + 60 + 27 + 1 + 3 = 95
        """
        return sum(
            RelationObservationLayout.block_dim(b, N, K)
            for b in RelationObservationLayout.BLOCK_ORDER
        )

    @staticmethod
    def block_offset(block_name: str, N: int, K: int) -> tuple[int, int]:
        """``(start, end)`` indices of a block in the flattened observation."""
        if block_name not in RelationObservationLayout.BLOCK_ORDER:
            raise ValueError(f"Unknown block: {block_name}")
        start = 0
        for b in RelationObservationLayout.BLOCK_ORDER:
            d = RelationObservationLayout.block_dim(b, N, K)
            if b == block_name:
                return (start, start + d)
            start += d
        raise RuntimeError("unreachable")


# Stage-6 alias: the v4 six-block class is gone; external references to the
# name resolve to the v5 layout.
ObservationLayout = RelationObservationLayout


def slice_relation_block(
    obs: np.ndarray, block_name: str, N: int, K: int,
) -> np.ndarray:
    """Slice a single v5 block out of a flattened observation (debug / viz)."""
    start, end = RelationObservationLayout.block_offset(block_name, N, K)
    return obs[..., start:end]


def pad_resource_block(
    visible: Iterable[tuple[float, float, float]],
    K: int,
) -> np.ndarray:
    """Pad a list of resource cells ``(rel_dx, rel_dy, q_norm)`` to ``K``.

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
    """Pad neighbors to ``N-1`` slots with a presence flag.

    Each ``visible`` entry is ``(rel_dx, rel_dy, last_action_onehot[6], present)``.
    Returns ``((N-1) * 9,)`` flattened float32 array.
    """
    slots = N - 1
    out = np.zeros(
        (slots, RelationObservationLayout.NEIGHBOR_PER_ITEM), dtype=np.float32,
    )
    for i, (dx, dy, act_oh, presence) in enumerate(visible):
        if i >= slots:
            break
        out[i, 0] = dx
        out[i, 1] = dy
        out[i, 2:8] = np.asarray(act_oh, dtype=np.float32)
        out[i, 8] = 1.0 if presence else 0.0
    return out.reshape(-1)
