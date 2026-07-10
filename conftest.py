"""Repo-root pytest config — implements the ``--runslow`` gate.

Every slow smoke test (``@pytest.mark.slow``: the 20K env-step external
baseline gates) has always documented "default ``pytest`` does not run this
unless ``--runslow`` is passed", but the hook implementing that gate was
never committed — so default full-suite runs silently trained MAPPO / QMIX /
MA-MuZero-GH for 3 seeds x 20K steps each. This conftest makes the documented
contract real: slow tests are collected but skipped unless ``--runslow``.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run @pytest.mark.slow tests (20K env-step external smoke gates)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: 20K env-step external baseline smoke gate; skipped unless --runslow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
