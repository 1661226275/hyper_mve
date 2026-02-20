"""
Replay Buffer for multi-agent environment.

Stores transitions as joint observations/actions.
Handles agents with different observation dimensions by padding.
"""
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer storing joint transitions.

    Each transition: (joint_obs, joint_action, joint_reward, joint_obs_next, done)
    where joint_obs is the concatenation of all agents' observations.
    """

    def __init__(self, cfg, device):
        self.buffer_size = cfg.buffer_size
        self.batch_size = cfg.batch_size
        self.device = device
        self.count = 0
        self.current_size = 0

        joint_obs_dim = cfg.joint_obs_dim
        joint_action_dim = cfg.joint_action_dim

        self.buffer_obs = np.empty((self.buffer_size, joint_obs_dim), dtype=np.float32)
        self.buffer_action = np.empty((self.buffer_size, joint_action_dim), dtype=np.float32)
        self.buffer_reward = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.buffer_obs_next = np.empty((self.buffer_size, joint_obs_dim), dtype=np.float32)
        self.buffer_done = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.buffer_rule = np.empty((self.buffer_size, 1), dtype=np.float32)

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n, rule=0.0):
        """
        Store a single transition.

        Args:
            obs_n:      list of np.ndarray, per-agent observations
            a_n:        list of np.ndarray, per-agent actions
            r_n:        list of float, per-agent rewards
            obs_next_n: list of np.ndarray, per-agent next observations
            done_n:     list of bool, per-agent done flags
            rule:       float, current environment rule
        """
        joint_obs = np.concatenate(obs_n)
        joint_action = np.concatenate(a_n)
        joint_reward = np.sum(r_n)  # sum of all agents' rewards
        joint_obs_next = np.concatenate(obs_next_n)
        done = float(all(done_n))

        self.buffer_obs[self.count] = joint_obs
        self.buffer_action[self.count] = joint_action
        self.buffer_reward[self.count] = joint_reward
        self.buffer_obs_next[self.count] = joint_obs_next
        self.buffer_done[self.count] = done
        self.buffer_rule[self.count] = rule

        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        """
        Sample a batch of transitions.

        Returns:
            batch_obs:      (batch_size, joint_obs_dim)
            batch_action:   (batch_size, joint_action_dim)
            batch_reward:   (batch_size, 1)
            batch_obs_next: (batch_size, joint_obs_dim)
            batch_done:     (batch_size, 1)
            batch_rule:     (batch_size, 1)
        """
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs = torch.tensor(self.buffer_obs[index], dtype=torch.float32).to(self.device)
        batch_action = torch.tensor(self.buffer_action[index], dtype=torch.float32).to(self.device)
        batch_reward = torch.tensor(self.buffer_reward[index], dtype=torch.float32).to(self.device)
        batch_obs_next = torch.tensor(self.buffer_obs_next[index], dtype=torch.float32).to(self.device)
        batch_done = torch.tensor(self.buffer_done[index], dtype=torch.float32).to(self.device)
        batch_rule = torch.tensor(self.buffer_rule[index], dtype=torch.float32).to(self.device)

        return batch_obs, batch_action, batch_reward, batch_obs_next, batch_done, batch_rule

    def ready(self):
        """Check if buffer has enough samples for a batch."""
        return self.current_size >= self.batch_size
