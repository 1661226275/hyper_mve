"""matplotlib RGB-array renderer for ResourceCommons (Pkg-02 spec 08 §3.4).

Uses ``buffer_rgba()`` (matplotlib ≥ 3.0, stable across versions) instead of
the deprecated ``tostring_rgb()``. Output is ``(H, W, 3) uint8``.
"""
from __future__ import annotations

import numpy as np

from hyper_mve.schemas._constants import Q_MAX

from .state import ResourceCommonsState


def render_rgb_array(state: ResourceCommonsState, L: int) -> np.ndarray:
    """Render the current state to an RGB image (for paper Fig 3.1 / debug)."""
    # Lazy-imported so importing the env module does not require a display.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=64)

    # Resource cells, colour-coded by current stock fraction.
    for k in range(state.resource_positions.shape[0]):
        x, y = state.resource_positions[k]
        q_norm = float(state.resource_stocks[k]) / float(Q_MAX)
        ax.scatter(x, y, s=100, c=[plt.cm.viridis(q_norm)], marker="o")

    # Agents, coloured by type (α=red, β=blue).
    for i in range(state.agent_positions.shape[0]):
        x, y = state.agent_positions[i]
        color = "red" if int(state.agent_types[i]) == 0 else "blue"
        ax.scatter(x, y, s=200, c=color, marker="s", edgecolors="black")

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.grid(True)
    ax.set_title(
        f"c_t={state.c_t:.2f}, step={state.step_idx}",
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())   # (H, W, 4) uint8
    img = rgba[..., :3].copy()                    # drop alpha; own the buffer
    plt.close(fig)
    return img
