"""Vendored from configs/dreamer/DreamerAgentConfig.py + DreamerLearnerConfig.py
+ DreamerControllerConfig.py (values folded into one flat config class).

[PORT] Changes vs upstream:
  * one flat ``MambaConfig`` instead of the Config class hierarchy;
  * ``ENV_TYPE`` removed (STARCRAFT semantics assumed throughout);
  * ``DEVICE`` defaults to cuda-if-available (upstream: 'cpu');
  * ``LOG_FOLDER`` removed (wandb removed).
Upstream numeric defaults preserved exactly (SMAC-flavoured base config).
"""
from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

RSSM_STATE_MODE = 'discrete'


class MambaConfig:
    def __init__(self):
        # --- DreamerConfig (configs/dreamer/DreamerAgentConfig.py) ---
        self.HIDDEN = 256
        self.MODEL_HIDDEN = 256
        self.EMBED = 256
        self.N_CATEGORICALS = 32
        self.N_CLASSES = 32
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 256
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        self.GLOBAL_FEAT = self.FEAT + self.EMBED
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 256
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 256
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 256
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 256
        self.GAMMA = 0.99
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95
        self.IN_DIM = 30

        # --- DreamerLearnerConfig ---
        # [PORT 2026-07-10] values from configs/dreamer/optimal/starcraft/
        # LearnerConfig.py — the README's "Optimal parameters" section requires
        # manually copying these over the base defaults; the base defaults
        # (MODEL_EPOCHS=1, EPOCHS=1, SEQ_LENGTH=50, MIN_BUFFER_SIZE=100,
        # CAPACITY=500000) leave the world model badly undertrained (verified:
        # 20k-step smoke stayed at random-play return).
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 500
        self.MODEL_EPOCHS = 60
        self.EPOCHS = 4
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 40
        self.BATCH_SIZE = 40
        self.SEQ_LENGTH = 20
        self.N_SAMPLES = 1
        self.GRAD_CLIP = 100.0
        self.HORIZON = 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.

        # --- DreamerControllerConfig ---
        self.EXPL_DECAY = 0.9999
        self.EXPL_NOISE = 0.
        self.EXPL_MIN = 0.

        # [PORT] upstream DreamerLearnerConfig.DEVICE = 'cpu'; we run in-process
        # on the sweep worker's (already CUDA_VISIBLE_DEVICES-masked) GPU.
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]
