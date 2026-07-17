"""matplotlib RGB-array renderer for RelationCommons (Pkg-09).

Adapted from the v4 renderer: hotspots and type coloring are gone; agents
are colored by id and the title shows the current regime id.
"""
from __future__ import annotations

import numpy as np

from hyper_mve.utils.schemas._constants import Q_MAX

from .state import RelationCommonsState


def render_rgb_array(state: RelationCommonsState, L: int) -> np.ndarray:
    """Render the current state to an RGB image ``(H, W, 3) uint8``."""
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

    # Agents, coloured by id.
    N = state.agent_positions.shape[0]
    for i in range(N):
        x, y = state.agent_positions[i]
        ax.scatter(
            x, y, s=200, c=[plt.cm.tab10(i % 10)], marker="s", edgecolors="black",
        )

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.grid(True)
    ax.set_title(f"g={state.g}, step={state.step_idx}")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())   # (H, W, 4) uint8
    img = rgba[..., :3].copy()                    # drop alpha; own the buffer
    plt.close(fig)
    return img
