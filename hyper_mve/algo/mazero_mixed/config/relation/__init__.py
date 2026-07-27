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

        # 2026-07-27 capacity control for the parameter-matched comparison.
        # --model_scale multiplies every width above (and the hypernet/context
        # widths in ModelConfig below), so a higher-parameter variant of the
        # method is a config change rather than a second model definition.
        # scale=1.0 is bit-identical to the numbers above (no-op by construction:
        # int(128*1.0)==128), so existing runs stay reproducible.
        scale = float(getattr(self, "model_scale", 1.0) or 1.0)
        if scale != 1.0:
            _w = lambda v: max(1, int(round(v * scale)))
            self.hidden_state_size = _w(self.hidden_state_size)
            self.fc_representation_layers = [_w(v) for v in self.fc_representation_layers]
            self.fc_dynamic_layers = [_w(v) for v in self.fc_dynamic_layers]
            self.fc_reward_layers = [_w(v) for v in self.fc_reward_layers]
            self.fc_value_layers = [_w(v) for v in self.fc_value_layers]
            self.fc_policy_layers = [_w(v) for v in self.fc_policy_layers]
            self.proj_hid = _w(self.proj_hid)
            self.proj_out = _w(self.proj_out)
            self.pred_hid = _w(self.pred_hid)
            self.pred_out = _w(self.pred_out)

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
                model_cfg=self._scaled_model_cfg(),
                n_regimes=5,
                belief_point_estimate=getattr(self, "belief_point_estimate", False),
                belief_blind=getattr(self, "belief_blind", False),
                value_hard_select=getattr(self, "value_hard_select", False),
                belief_grad_gating_steps=getattr(self, "belief_grad_gating_steps", 5000),
                conditioning=getattr(self, "conditioning", "hyper"),
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

    def _scaled_model_cfg(self):
        """ModelConfig with the subjective/hypernet widths scaled by model_scale.

        The dimension fields with hard architectural constraints (d_role,
        d_belief, d_id_emb, d_row_emb — checked in ModelConfig.__post_init__)
        are left alone; only the free capacity knobs scale, so a scaled config
        still satisfies those constraints.
        """
        from dataclasses import replace
        from hyper_mve.utils.configs import ModelConfig

        cfg = ModelConfig()
        scale = float(getattr(self, "model_scale", 1.0) or 1.0)
        if scale == 1.0:
            return cfg
        _w = lambda v: max(1, int(round(v * scale)))
        return replace(
            cfg,
            latent_dim=_w(cfg.latent_dim),
            hidden_dim=_w(cfg.hidden_dim),
            hyper_hidden_dims=tuple(_w(v) for v in cfg.hyper_hidden_dims),
            hyper_rew_hidden_dims=tuple(_w(v) for v in cfg.hyper_rew_hidden_dims),
        )

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
