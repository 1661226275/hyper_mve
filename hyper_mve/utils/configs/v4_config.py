"""V4Config — composes the 5-layer config + presets entry point."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .baselines_config import BaselinesConfig
from .env_config import EnvConfig
from .eval_config import EvalConfig
from .legacy_config import LegacyConfig
from .model_config import ModelConfig
from .mup_config import MupConfig
from .train_config import TrainConfig


_PRESET_NAMES: tuple[str, ...] = (
    "rel_duo", "rel_duo_holdout", "mpe_tag", "mpe_tag_fixed",
)


@dataclass(frozen=True)
class V4Config:
    """Top-level configuration (v5 rel_* presets fill the sub-configs).

    Use ``V4Config.from_preset("rel_duo" | "rel_duo_holdout")`` for standard
    experiments; use ``dataclasses.replace`` for explicit local overrides
    (Decision D8).
    """

    env: EnvConfig
    model: ModelConfig
    train: TrainConfig
    mup: MupConfig = field(default_factory=MupConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

    preset_name: str = "custom"

    @classmethod
    def from_preset(cls, name: str) -> "V4Config":
        """Load one of the reference configurations."""
        if name == "rel_duo":
            from .presets.rel_duo import build_rel_duo_config
            return build_rel_duo_config()
        if name == "rel_duo_holdout":
            from .presets.rel_duo import build_rel_duo_holdout_config
            return build_rel_duo_holdout_config()
        if name == "mpe_tag":
            from .presets.mpe_tag import build_mpe_tag_config
            return build_mpe_tag_config()
        if name == "mpe_tag_fixed":
            from .presets.mpe_tag import build_mpe_tag_fixed_config
            return build_mpe_tag_fixed_config()
        raise ValueError(
            f"Unknown preset: {name!r}. Valid: {_PRESET_NAMES}"
        )

    def to_dict(self) -> dict:
        """JSON-serialisable dict (tuples → lists)."""
        return asdict(self)
