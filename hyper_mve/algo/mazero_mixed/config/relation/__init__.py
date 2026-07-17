"""MAZero GameConfig for the RelationCommons relationship-regime env.

env_name variants:
  * ``rel_duo_coop`` — regime family restricted to g0 (mutual_coop): the
    stage-1 pristine-base smoke setting (cooperative MAZero must learn here).
  * ``rel_duo``      — full 5-regime family, regimes resampled per episode
    and hidden (the mixed-game target setting; used from stage 3 on).

The repo root is inserted into ``sys.path`` so ``hyper_mve.*`` imports work
when main.py is launched from the fork directory (mirrors how the repo's
pytest rootdir conftest exposes the package).
"""
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataclasses import replace

from core.config import BaseConfig, DiscreteSupport

from .model import MAMuZeroNet
from .env_wrapper import RelationCommonsGame


class GameConfig(BaseConfig):
    def __init__(self, args):
        super(GameConfig, self).__init__(args)

        self.hidden_state_size = 128
        self.fc_representation_layers = [128, 128]
        self.fc_dynamic_layers = [128, 128]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]
        self.proj_hid = 128
        self.proj_out = 128
        self.pred_hid = 64
        self.pred_out = 128

        if self.use_vectorization:
            # Per-step team reward is a mean of relational rewards (|r| ≲ 3);
            # values pass through the MuZero invertible scaling transform.
            self.value_support = DiscreteSupport(-20, 20)
            self.reward_support = DiscreteSupport(-5, 5)

    def get_uniform_network(self):
        if getattr(self, "subjective_model", False):
            from hyper_mve.utils.configs import ModelConfig
            from .subjective_model import HyperMAMuZeroNet

            env_cfg = self._make_env_cfg()
            assert not self.use_vectorization, (
                "subjective_model currently supports the scalar value transform "
                "only (functional heads emit 1-dim outputs)."
            )
            assert self.stacked_observations == 1, (
                "subjective_model extracts rows/belief from the single current "
                "observation; stacked_observations must be 1."
            )
            return HyperMAMuZeroNet(
                self.num_agents,
                (self.stacked_observations, *self.obs_shape),
                self.action_space_size,
                self.hidden_state_size,
                self.fc_representation_layers,
                self.fc_dynamic_layers,
                self.fc_policy_layers,
                self.inverse_value_transform,
                self.inverse_reward_transform,
                env_cfg=env_cfg,
                model_cfg=ModelConfig(),
                n_regimes=5,
                belief_point_estimate=getattr(self, "belief_point_estimate", False),
                belief_grad_gating_steps=getattr(self, "belief_grad_gating_steps", 5000),
                proj_hid=self.proj_hid, proj_out=self.proj_out,
                pred_hid=self.pred_hid, pred_out=self.pred_out,
                use_feature_norm=True,
            )
        return MAMuZeroNet(
            self.num_agents,
            (self.stacked_observations, *self.obs_shape),
            self.action_space_size,
            self.hidden_state_size,
            self.fc_representation_layers,
            self.fc_dynamic_layers,
            self.fc_reward_layers,
            self.fc_value_layers,
            self.fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            use_feature_norm=True,
        )

    def _make_env_cfg(self):
        from hyper_mve.utils.configs import V4Config

        # Suite-integration hook: an ExternalBaselineRunner can pin the exact
        # EnvConfig from the harness cfg (overrides the env_name presets).
        override = getattr(self, "env_cfg_override", None)
        if override is not None:
            return override

        if self.env_name in ("mpe_tag", "mpe_tag_fixed"):
            return V4Config.from_preset(self.env_name).env

        base = V4Config.from_preset("rel_duo")
        env_cfg = base.env
        if self.env_name == "rel_duo_coop":
            env_cfg = replace(env_cfg, train_regime_ids=(0,))  # mutual_coop only
        elif self.env_name == "rel_duo":
            pass  # full hidden 5-regime family
        else:
            raise ValueError(
                f"Unknown relation env_name {self.env_name!r}; "
                "expected 'rel_duo', 'rel_duo_coop', 'mpe_tag' or 'mpe_tag_fixed'."
            )
        return env_cfg

    def new_game(self, seed=None, oracle=False, **kwargs):
        from hyper_mve.envs.adapters.pettingzoo_wrapper import (
            RelationCommonsPettingZooEnv,
        )

        env_cfg = self._make_env_cfg()
        # oracle=True (self-play data collection only): g_true flows into info
        # for the belief supervision signal — train-time privileged info per
        # CTDE; evaluation games never get it.
        if getattr(env_cfg, "env_kind", "relation") == "mpe_tag":
            from hyper_mve.envs.mpe_tag import MPETagRegimeEnv

            env = MPETagRegimeEnv(
                env_cfg, oracle_mode=bool(oracle), eval_info_mode=False
            )
        else:
            env = RelationCommonsPettingZooEnv(
                env_cfg, oracle_mode=bool(oracle), eval_info_mode=False
            )
        game = RelationCommonsGame(env, T_max=env_cfg.T_max)
        if seed is not None:
            game.set_seed(seed)
        return game
