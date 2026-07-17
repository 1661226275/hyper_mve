"""Lock 3 — matplotlib Agg backend; render_plot writes non-empty PNG (pkg-08 spec 07 §8 / §11.7)."""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_compare_module_uses_agg_backend():
    """Lock 3 — matplotlib backend is 'agg' after importing compare module."""
    # Force re-import to make sure the use("Agg") line ran in this process.
    import matplotlib
    import hyper_mve.utils.analysis.compare  # noqa: F401
    importlib.reload(hyper_mve.utils.analysis.compare)
    assert matplotlib.get_backend().lower() == "agg"


def test_render_plot_produces_nonempty_png(tmp_path):
    """spec 07 §11.7 — PNG file exists and is non-empty."""
    from hyper_mve.utils.analysis.compare import render_plot
    from hyper_mve.utils.analysis.stats import compare_methods

    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=5),
        "input_wide": rng.normal(14.1, 0.4, size=5),
    }
    result = compare_methods(method_returns, alpha=0.05)
    out_path = tmp_path / "compare_return_mean.png"
    render_plot(result, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_plot_closes_figures(tmp_path):
    """spec 07 §8.3 — render_plot does not leak figure handles."""
    import matplotlib.pyplot as plt
    from hyper_mve.utils.analysis.compare import render_plot
    from hyper_mve.utils.analysis.stats import compare_methods

    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=5),
        "input_wide": rng.normal(14.1, 0.4, size=5),
    }
    result = compare_methods(method_returns, alpha=0.05)
    plt.close("all")
    for i in range(8):
        render_plot(result, tmp_path / f"p{i}.png")
    assert plt.get_fignums() == []
