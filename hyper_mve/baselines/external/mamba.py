"""MAMBAAlgorithm — Tier-2 stub (pkg-07 spec 06 §4 + design D8).

Per the user's Phase 0 decision (option "Stub from day 1"), the design D8
2-day sourcing window is **not** opened. The spec 06 §4.6 mechanism is
**class binding at import-time**:

    MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub

When ``IS_SOURCED is False`` (this pass), ``MAMBAAlgorithm`` resolves to
``_MAMBAStub``, whose ``__init__`` raises ``NotImplementedError`` so the
factory call ``create_baseline(cfg, "external_mamba")`` fails loudly.
When/if MAMBA is sourced (a future spec 06 §4.4 amendment), the
implementer flips ``IS_SOURCED = True`` and authors ``_RealMAMBA`` in
this file; the registry binding picks up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Final, Union

from hyper_mve.baselines.external.base import ExternalBaselineRunner
from hyper_mve.configs import V4Config


#: Module-level toggle (pkg-07 spec 06 §4.6 + spec 01 §3.2 line 139).
#: ``False`` → :class:`_MAMBAStub`; ``True`` → :class:`_RealMAMBA`.
IS_SOURCED: Final[bool] = False


class _MAMBAStub(ExternalBaselineRunner):
    """Stub fallback when MAMBA sourcing is not opened (§4.5)."""

    def __init__(self, cfg: V4Config) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "MAMBA Tier-2: sourcing window not opened (pkg-07 design D8). "
            "Per spec 06 §4.6 the runtime fallback is _MAMBAStub raising on "
            "__init__; flip hyper_mve.baselines.external.mamba.IS_SOURCED to "
            "True after porting the implementation."
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
        raise NotImplementedError("MAMBA stub — see __init__.")

    def evaluate(self, env_fn, regime_grid, episodes):
        raise NotImplementedError("MAMBA stub — see __init__.")

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        raise NotImplementedError("MAMBA stub — see __init__.")

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        raise NotImplementedError("MAMBA stub — see __init__.")


class _RealMAMBA(ExternalBaselineRunner):
    """Vendored MAMBA impl, written at sourcing time per spec 06 §4.4.

    This pass: no body. ``IS_SOURCED`` is ``False``, so this class is not
    bound to ``MAMBAAlgorithm``.
    """

    def __init__(self, cfg: V4Config) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "_RealMAMBA: body deferred to a future spec 06 §4.4 amendment."
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
        raise NotImplementedError("MAMBA real port not yet authored.")


# pkg-07 spec 06 §4.6 byte-identical class-binding:
MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub


__all__ = ["MAMBAAlgorithm", "IS_SOURCED"]
