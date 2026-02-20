"""
Utility functions for Hyper-MuZero.

Includes:
    - Reward/Value invertible scaling (MuZero-style)
    - Orthogonal initialization
    - Device selection
"""
import torch
import torch.nn as nn
import numpy as np


# ── Reward/Value Scaling ─────────────────────────────────────────
# h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
# h_inv(x) = sign(x) * ((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps)) ** 2 - 1)

def scalar_transform(x, eps=0.001):
    """
    MuZero invertible scaling: h(x) = sign(x)(sqrt(|x|+1)-1) + eps*x

    Args:
        x: tensor of any shape
        eps: small constant for invertibility
    Returns:
        scaled tensor of same shape
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def inverse_scalar_transform(x, eps=0.001):
    """
    Inverse of scalar_transform: h^{-1}(x)

    Args:
        x: scaled tensor
        eps: must match the eps used in scalar_transform
    Returns:
        unscaled tensor
    """
    # Solve h(y) = x for y
    # y = sign(x) * ( ((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
    sign_x = torch.sign(x)
    abs_x = torch.abs(x)
    inner = torch.sqrt(1.0 + 4.0 * eps * (abs_x + 1.0 + eps))
    y = sign_x * (((inner - 1.0) / (2.0 * eps)) ** 2 - 1.0)
    return y


# ── Initialization ──────────────────────────────────────────────

def orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization for linear layers."""
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


def small_init(layer, std=0.01):
    """Small weight initialization for output layers (e.g. HyperNet output)."""
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.normal_(param, mean=0.0, std=std)


# ── Device ──────────────────────────────────────────────────────

def get_device(device_str='auto'):
    """
    Get torch device.

    Args:
        device_str: 'auto', 'cpu', or 'cuda'
    Returns:
        torch.device
    """
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


# ── Action encoding ─────────────────────────────────────────────

# ── Adversarial Freeze (Phase Scheduling) ─────────────────────

def get_active_agents(train_steps, cfg):
    """
    Determine which agents are active for perspective sampling
    based on current training step and freeze config.

    Phase schedule (when freeze_enabled=True):
        - steps < freeze_warmup_steps: all agents (warmup)
        - Even phases (0, 2, 4...): hunters only (cfg.freeze_hunter_agents)
        - Odd phases (1, 3, 5...):  prey only (cfg.freeze_prey_agents)

    Args:
        train_steps: int, current total training steps
        cfg: BaseConfig with freeze_* attributes

    Returns:
        active_agents: list of int or None (None = all agents)
        phase_name: str for logging ('All', 'Hunters', 'Prey')
    """
    if not getattr(cfg, 'freeze_enabled', False):
        return None, 'All'

    warmup = getattr(cfg, 'freeze_warmup_steps', 5000)
    if train_steps < warmup:
        return None, 'Warmup'

    phase_steps = getattr(cfg, 'freeze_phase_steps', 10000)
    elapsed = train_steps - warmup
    phase_idx = elapsed // phase_steps

    if phase_idx % 2 == 0:
        agents = getattr(cfg, 'freeze_hunter_agents', [0, 1, 2])
        return agents, 'Hunters'
    else:
        agents = getattr(cfg, 'freeze_prey_agents', [3])
        return agents, 'Prey'


def actions_to_one_hot(actions, num_actions=5):
    """
    Convert integer actions to one-hot encoding and concatenate for joint action.

    Args:
        actions: (batch_size, num_agents) int tensor, values in [0, num_actions)
        num_actions: number of discrete actions per agent
    Returns:
        joint_action_onehot: (batch_size, num_agents * num_actions) float tensor
    """
    batch_size, num_agents = actions.shape
    # One-hot each agent's action
    one_hot = torch.zeros(batch_size, num_agents, num_actions,
                          device=actions.device, dtype=torch.float32)
    one_hot.scatter_(2, actions.unsqueeze(-1).long(), 1.0)
    # Flatten: (B, N, A) -> (B, N*A)
    return one_hot.view(batch_size, num_agents * num_actions)
