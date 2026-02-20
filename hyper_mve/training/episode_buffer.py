"""
Episode Replay Buffer for MuZero-style training.

Stores complete episodes and supports sampling contiguous sequences
of length K+1 for unrolled training.

Storage per episode:
    - obs:              (T+1, joint_obs_dim)   joint observations (includes final obs)
    - actions:          (T, num_agents)        discrete actions (int)
    - rewards:          (T, num_agents)        per-agent rewards (float)
    - search_policies:  (T, num_agents, num_actions)  MVE planner output
    - rule:             scalar                 episode Rule value
    - dones:            (T,)                   done flags
    - length:           int (= T)              actual episode length (number of actions)

v4.4: Removed PER (Priority Experience Replay), switched to uniform sampling.
    Reason: reward-magnitude PER causes stale high-reward episodes to be over-sampled,
    conflicting with buffer_size reduction (5000) for data freshness.
"""
import numpy as np
import torch
from collections import deque


class EpisodeData:
    """Container for a single episode's data."""

    def __init__(self, obs, actions, rewards, search_policies, rule, dones, length):
        """
        Args:
            obs:              np.ndarray (T, joint_obs_dim)
            actions:          np.ndarray (T, num_agents) int
            rewards:          np.ndarray (T, num_agents) float
            search_policies:  np.ndarray (T, num_agents, num_actions) float
            rule:             float
            dones:            np.ndarray (T,) bool
            length:           int
        """
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.search_policies = search_policies
        self.rule = rule
        self.dones = dones
        self.length = length


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for MuZero unrolled training.

    Stores complete episodes and samples contiguous sequences of length unroll_K + 1.
    Uses uniform sampling (v4.4) to maximize data freshness with small buffer.
    """

    def __init__(self, cfg, device):
        """
        Args:
            cfg: config with buffer_size, batch_size, unroll_K, num_agents, num_actions, joint_obs_dim
            device: torch device
        """
        self.buffer_size = cfg.buffer_size  # max number of episodes
        self.batch_size = cfg.batch_size
        self.unroll_K = cfg.unroll_K
        self.num_agents = cfg.num_agents
        self.num_actions = cfg.num_actions
        self.joint_obs_dim = cfg.joint_obs_dim
        self.device = device

        self.buffer = deque(maxlen=self.buffer_size)

    def store_episode(self, episode_data):
        """
        Store a complete episode. Episodes shorter than unroll_K are rejected.

        Args:
            episode_data: EpisodeData instance
        """
        if episode_data.length > self.unroll_K:
            self.buffer.append(episode_data)

    def sample_batch(self):
        """
        Sample a batch with uniform random sampling.

        Each sample: a random episode, random start position t,
        yielding sequence [t, t+K] (K+1 observations, K actions/rewards/policies).

        Returns:
            dict with keys:
                'obs':      (B, K+1, joint_obs_dim)  float32
                'actions':  (B, K, num_agents)        int64
                'rewards':  (B, K, num_agents)        float32
                'policies': (B, K, num_agents, num_actions) float32
                'rules':    (B,)                      float32
                'dones':    (B, K)                    float32
        """
        B = self.batch_size
        K = self.unroll_K

        obs_batch = np.zeros((B, K + 1, self.joint_obs_dim), dtype=np.float32)
        actions_batch = np.zeros((B, K, self.num_agents), dtype=np.int64)
        rewards_batch = np.zeros((B, K, self.num_agents), dtype=np.float32)
        policies_batch = np.zeros((B, K, self.num_agents, self.num_actions), dtype=np.float32)
        rules_batch = np.zeros(B, dtype=np.float32)
        dones_batch = np.zeros((B, K), dtype=np.float32)

        ep_indices = np.random.choice(len(self.buffer), size=B)

        for i, ep_idx in enumerate(ep_indices):
            ep = self.buffer[ep_idx]
            # Random start position ensuring t + K <= length - 1
            # We need K+1 obs: [t, t+1, ..., t+K]
            # and K actions/rewards: [t, t+1, ..., t+K-1]
            max_start = ep.length - K
            t = np.random.randint(0, max_start)

            obs_batch[i] = ep.obs[t:t + K + 1]
            actions_batch[i] = ep.actions[t:t + K]
            rewards_batch[i] = ep.rewards[t:t + K]
            policies_batch[i] = ep.search_policies[t:t + K]
            rules_batch[i] = ep.rule
            dones_batch[i] = ep.dones[t:t + K].astype(np.float32)

        return {
            'obs': torch.tensor(obs_batch, dtype=torch.float32).to(self.device),
            'actions': torch.tensor(actions_batch, dtype=torch.long).to(self.device),
            'rewards': torch.tensor(rewards_batch, dtype=torch.float32).to(self.device),
            'policies': torch.tensor(policies_batch, dtype=torch.float32).to(self.device),
            'rules': torch.tensor(rules_batch, dtype=torch.float32).to(self.device),
            'dones': torch.tensor(dones_batch, dtype=torch.float32).to(self.device),
        }

    def sample_batch_with_history(self, trajectory_window=10):
        """
        Sample batch with additional history window for GRU context inference.

        Returns everything from sample_batch() plus:
            'history_obs':            (B, W, joint_obs_dim)
            'history_actions_onehot': (B, W, joint_action_dim)
            'history_rewards':        (B, W, 1)  — mean reward across agents
            'history_mask':           (B, W)      — True for valid, False for padding

        The history window covers [t-W, t-1] (steps before the unroll start).
        If t < W, the history is left-padded with zeros and mask marks invalid steps.

        Args:
            trajectory_window: W, number of history steps
        Returns:
            dict with all keys from sample_batch() plus history keys
        """
        B = self.batch_size
        K = self.unroll_K
        W = trajectory_window
        joint_action_dim = self.num_agents * self.num_actions

        obs_batch = np.zeros((B, K + 1, self.joint_obs_dim), dtype=np.float32)
        actions_batch = np.zeros((B, K, self.num_agents), dtype=np.int64)
        rewards_batch = np.zeros((B, K, self.num_agents), dtype=np.float32)
        policies_batch = np.zeros((B, K, self.num_agents, self.num_actions), dtype=np.float32)
        rules_batch = np.zeros(B, dtype=np.float32)
        dones_batch = np.zeros((B, K), dtype=np.float32)

        # History arrays
        hist_obs = np.zeros((B, W, self.joint_obs_dim), dtype=np.float32)
        hist_actions = np.zeros((B, W, joint_action_dim), dtype=np.float32)
        hist_rewards = np.zeros((B, W, 1), dtype=np.float32)
        hist_mask = np.zeros((B, W), dtype=bool)

        ep_indices = np.random.choice(len(self.buffer), size=B)

        for i, ep_idx in enumerate(ep_indices):
            ep = self.buffer[ep_idx]
            max_start = ep.length - K
            t = np.random.randint(0, max_start)

            # Standard batch data
            obs_batch[i] = ep.obs[t:t + K + 1]
            actions_batch[i] = ep.actions[t:t + K]
            rewards_batch[i] = ep.rewards[t:t + K]
            policies_batch[i] = ep.search_policies[t:t + K]
            rules_batch[i] = ep.rule
            dones_batch[i] = ep.dones[t:t + K].astype(np.float32)

            # History window: [t-W, t-1]
            hist_start = max(0, t - W)
            hist_len = t - hist_start  # actual number of valid history steps

            if hist_len > 0:
                # Fill from the right (most recent history at the end)
                pad_len = W - hist_len
                hist_obs[i, pad_len:] = ep.obs[hist_start:t]
                # One-hot encode actions for history
                for j in range(hist_len):
                    act_j = ep.actions[hist_start + j]  # (N,)
                    for a_idx in range(self.num_agents):
                        hist_actions[i, pad_len + j,
                                     a_idx * self.num_actions + act_j[a_idx]] = 1.0
                # Mean reward across agents
                hist_rewards[i, pad_len:, 0] = ep.rewards[hist_start:t].mean(axis=1)
                hist_mask[i, pad_len:] = True

        result = {
            'obs': torch.tensor(obs_batch, dtype=torch.float32).to(self.device),
            'actions': torch.tensor(actions_batch, dtype=torch.long).to(self.device),
            'rewards': torch.tensor(rewards_batch, dtype=torch.float32).to(self.device),
            'policies': torch.tensor(policies_batch, dtype=torch.float32).to(self.device),
            'rules': torch.tensor(rules_batch, dtype=torch.float32).to(self.device),
            'dones': torch.tensor(dones_batch, dtype=torch.float32).to(self.device),
            'history_obs': torch.tensor(hist_obs, dtype=torch.float32).to(self.device),
            'history_actions_onehot': torch.tensor(hist_actions, dtype=torch.float32).to(self.device),
            'history_rewards': torch.tensor(hist_rewards, dtype=torch.float32).to(self.device),
            'history_mask': torch.tensor(hist_mask, dtype=torch.bool).to(self.device),
        }
        return result

    def ready(self):
        """Check if buffer has enough episodes for a batch."""
        return len(self.buffer) >= self.batch_size

    def __len__(self):
        return len(self.buffer)
