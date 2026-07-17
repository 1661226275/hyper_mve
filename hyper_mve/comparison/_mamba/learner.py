"""Vendored from agent/learners/DreamerLearner.py.

[PORT] changes:
  * relative imports; wandb.init/log removed (``metrics_cb`` hook instead,
    forwarded to ``loss.LOGGER``);
  * ``orthogonal_init``'s manual ``torch.svd`` path replaced with
    ``torch.nn.init.orthogonal_`` (torch.svd is deprecated in torch 2.x;
    identical intent — orthogonal init with gain);
  * FLATLAND-only ``old_critic`` / ``TARGET_UPDATE`` machinery removed;
    ``train_agent`` keeps the upstream STARCRAFT path (``critic=self.critic``
    in actor_rollout, ``advantage()`` normalisation);
  * ``DreamerMemory`` constructed without the env_type arg (see memory.py).
"""
import sys

import numpy as np
import torch

from .memory import DreamerMemory            # [PORT] was: agent.memory.DreamerMemory
from .model import DreamerModel              # [PORT] was: agent.models.DreamerModel
from . import loss as loss_module            # [PORT] LOGGER hook lives here
from .loss import model_loss, actor_loss, value_loss, actor_rollout
from .optim_utils import advantage           # [PORT] was: agent.optim.utils
from .action import Actor                    # [PORT] was: networks.dreamer.action
from .critic import AugmentedCritic          # [PORT] was: networks.dreamer.critic


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                # [PORT] upstream used a hand-rolled torch.svd orthogonal init;
                # torch.nn.init.orthogonal_ implements the same initialisation.
                torch.nn.init.orthogonal_(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:

    def __init__(self, config, metrics_cb=None):
        self.config = config
        self.model = DreamerModel(config).to(config.DEVICE).eval()
        self.actor = Actor(config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(
            config.DEVICE)
        self.critic = AugmentedCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        # [PORT] wandb.init removed; optional metrics callback instead.
        self.metrics_cb = metrics_cb
        loss_module.LOGGER = metrics_cb

    def init_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def params(self):
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples)

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            self.train_agent(samples)

    def train_model(self, samples):
        self.model.train()
        loss = model_loss(self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'], samples['last'])
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()

    def train_agent(self, samples):
        # [PORT] upstream STARCRAFT path: critic=self.critic + advantage();
        # FLATLAND old_critic/TARGET_UPDATE branches removed.
        actions, av_actions, old_policy, imag_feat, returns = actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'], self.model,
                                                                            self.actor,
                                                                            self.critic,
                                                                            self.config)
        adv = returns.detach() - self.critic(imag_feat).detach()
        adv = advantage(adv)
        if self.metrics_cb is not None:
            self.metrics_cb({'Agent/Returns': returns.mean()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, imag_feat[idx], returns[idx])
                if np.random.randint(20) == 9 and self.metrics_cb is not None:
                    self.metrics_cb({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
