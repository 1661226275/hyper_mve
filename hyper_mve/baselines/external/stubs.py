"""Permanent stubs: MARIE + GA (pkg-07 design D9).

Both classes raise ``NotImplementedError`` on construction. They exist to
preserve CLI/registry slots so the methods main table can keep the columns
visible without actually shipping training code.

pkg-07 spec 06 §5.2: MARIEStub and GAStub are **separate classes** (not a
shared base instance) so that ``isinstance(runner, MARIEStub) vs
isinstance(runner, GAStub)`` is a meaningful check downstream. The shared
implementation lives on :class:`_PermanentStubBase`; the concrete classes
only override ``_STUB_NAME``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

from hyper_mve.baselines.external.base import ExternalBaselineRunner
from hyper_mve.configs import V4Config


class _PermanentStubBase(ExternalBaselineRunner):
    """Common base for permanent stubs."""

    _STUB_NAME: str = "UNDEFINED"
    _SPEC_REF: str = "pkg-07 design D9"

    def __init__(self, cfg: V4Config) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            f"{self._STUB_NAME}: permanent stub ({self._SPEC_REF}). "
            "CLI/registry slot preserved; no implementation planned."
        )

    def train(
        self,
        cfg: V4Config,
        env_fn: Callable[[], Any],
        *,
        total_env_steps: int = 0,
        lr: float = 0.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(f"{self._STUB_NAME} stub — see __init__.")

    def evaluate(self, env_fn, c_grid, episodes):
        raise NotImplementedError(f"{self._STUB_NAME} stub — see __init__.")

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        raise NotImplementedError(f"{self._STUB_NAME} stub — see __init__.")

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        raise NotImplementedError(f"{self._STUB_NAME} stub — see __init__.")


class MARIEStub(_PermanentStubBase):
    _STUB_NAME = "MARIE"


class GAStub(_PermanentStubBase):
    _STUB_NAME = "GA"


__all__ = ["MARIEStub", "GAStub"]
