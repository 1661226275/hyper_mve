"""V4Config — composes the 5-layer config + presets entry point."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from hyper_mve.schemas import AgentType

from .baselines_config import BaselinesConfig
from .env_config import EnvConfig
from .eval_config import EvalConfig
from .legacy_config import LegacyConfig
from .model_config import ModelConfig
from .mup_config import MupConfig
from .train_config import TrainConfig


_PRESET_NAMES: tuple[str, ...] = (
    "easy", "medium", "hard", "duo", "duo_basegen",
    "duo_film_lora", "duo_film_lora_fc2", "duo_base_lora",
    "medium_film_lora", "medium_film_lora_fc2", "medium_base_lora",
)


@dataclass(frozen=True)
class V4Config:
    """Top-level v4 configuration (Ch3.9 presets fill the sub-configs).

    Use ``V4Config.from_preset("easy" | "medium" | "hard")`` for standard
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
        """Load one of the Ch3.9 reference configurations."""
        if name == "easy":
            from .presets.easy import build_easy_config
            return build_easy_config()
        if name == "medium":
            from .presets.medium import build_medium_config
            return build_medium_config()
        if name == "hard":
            from .presets.hard import build_hard_config
            return build_hard_config()
        if name == "duo":
            from .presets.duo import build_duo_config
            return build_duo_config()
        if name == "duo_basegen":
            from .presets.duo_basegen import build_duo_basegen_config
            return build_duo_basegen_config()
        if name == "duo_film_lora":
            from .presets.duo_film_lora import build_duo_film_lora_config
            return build_duo_film_lora_config()
        if name == "duo_film_lora_fc2":
            from .presets.duo_film_lora_fc2 import build_duo_film_lora_fc2_config
            return build_duo_film_lora_fc2_config()
        if name == "duo_base_lora":
            from .presets.duo_base_lora import build_duo_base_lora_config
            return build_duo_base_lora_config()
        if name == "medium_film_lora":
            from .presets.medium_film_lora import build_medium_film_lora_config
            return build_medium_film_lora_config()
        if name == "medium_film_lora_fc2":
            from .presets.medium_film_lora_fc2 import build_medium_film_lora_fc2_config
            return build_medium_film_lora_fc2_config()
        if name == "medium_base_lora":
            from .presets.medium_base_lora import build_medium_base_lora_config
            return build_medium_base_lora_config()
        raise ValueError(
            f"Unknown preset: {name!r}. Valid: {_PRESET_NAMES}"
        )

    def to_dict(self) -> dict:
        """JSON-serialisable dict (AgentType → int, tuples → lists)."""
        d = asdict(self)
        d["env"]["type_assignment"] = [int(t) for t in self.env.type_assignment]
        return d
