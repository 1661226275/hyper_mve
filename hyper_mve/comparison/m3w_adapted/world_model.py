"""RegimeCondWorldModel — M3W's MoE world model, conditioned on a GIVEN
regime ID (the user-locked "M3W-adapted" spec).

Core mechanism preserved from the clone (imported UNMODIFIED via
``vendor_import``):
  * ``CenMoEDynamicsModel`` — SoftMoE latent dynamics (einsum dispatch/combine
    soft routing over MLP experts);
  * ``CenMoERewardModel``  — SparseMoE reward model (``NoisyTopKRouter`` +
    self-attention experts); ``n_r = n_agents`` gives PER-AGENT reward
    outputs (general-sum adaptation).

Our adaptation glue (this file): a weight-shared per-agent observation
encoder and ``nn.Embedding(n_regimes, d_g)`` — the given one-hot regime ID
is embedded and concatenated onto every agent latent, so both MoE modules
operate on the task-conditioned latent ``zc = [enc(o_i), e_g]``.

Losses: latent one-step consistency (stop-grad encoder target, TD-MPC
style), per-agent reward MSE, and the router's load-balancing loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendor_import import load_vendor_modules


class RegimeCondWorldModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        n_regimes: int,
        d_z: int = 64,
        d_g: int = 8,
        dyn_mlp_dims: tuple[int, ...] = (128,),
        n_dyn_experts: int = 4,
        n_rew_experts: int = 4,
        rew_top_k: int = 2,
        bal_coef: float = 0.01,
    ) -> None:
        super().__init__()
        CenMoEDynamicsModel, CenMoERewardModel, _ = load_vendor_modules()

        self.obs_dim = int(obs_dim)
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.n_regimes = int(n_regimes)
        self.d_z = int(d_z)
        self.d_g = int(d_g)
        self.d_zc = self.d_z + self.d_g
        self.rew_top_k = int(rew_top_k)
        self.bal_coef = float(bal_coef)

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 128), nn.ReLU(),
            nn.Linear(128, self.d_z),
        )
        self.g_embed = nn.Embedding(self.n_regimes, self.d_g)

        # Vendor modules are ALWAYS constructed with device="cpu" and moved
        # by the enclosing module's .to(): CenMoEDynamicsModel registers its
        # router as `nn.Parameter(...).to(device)`, which for a real device
        # move returns a plain (unregistered) tensor copy — the cpu
        # construction path is the only one that keeps `phi` a Parameter.
        self.dynamics = CenMoEDynamicsModel(
            d_z=self.d_zc, d_a=self.n_actions,
            mlp_dims=list(dyn_mlp_dims), n_experts=int(n_dyn_experts),
            device="cpu",
        )
        self.reward = CenMoERewardModel(
            d_z=self.d_zc, d_a=self.n_actions,
            n_agents=self.n_agents, n_experts=int(n_rew_experts),
            k=self.rew_top_k, n_r=self.n_agents, n_heads=1,
            expert_ffn_hidden=128, head_hidden=128,
            noisy_gating=True, device="cpu",
        )

    # --------------------------------------------------------------- encode
    def encode(self, obs: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """obs [B, N, obs_dim], g [B] long → conditioned latent [B, N, d_zc]."""
        B, N, _ = obs.shape
        z = self.encoder(obs)                                  # [B, N, d_z]
        e = self.g_embed(g.long()).unsqueeze(1).expand(B, N, self.d_g)
        return torch.cat([z, e], dim=-1)

    def _one_hot(self, a: torch.Tensor) -> torch.Tensor:
        return F.one_hot(a.long(), self.n_actions).float()

    # ------------------------------------------------------------- predict
    def predict_next(self, zc: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """zc [B, N, d_zc], a [B, N] long → next latent [B, N, d_zc]."""
        return self.dynamics.predict(zc, self._one_hot(a))

    def predict_rewards(self, zc: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """zc [B, N, d_zc], a [B, N] long → per-agent rewards [B, N]."""
        return self.reward.predict(zc, self._one_hot(a))

    # ----------------------------------------------------------------- loss
    def loss(
        self,
        obs: torch.Tensor, g: torch.Tensor, a: torch.Tensor,
        r: torch.Tensor, next_obs: torch.Tensor, next_g: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One-step WM losses on a real-transition batch (a [B,N] long,
        r [B,N] per-agent)."""
        zc = self.encode(obs, g)
        a1h = self._one_hot(a)
        with torch.no_grad():
            target = self.encode(next_obs, next_g)
        pred_next = self.dynamics.predict(zc, a1h)
        dyn_loss = F.mse_loss(pred_next, target)

        r_hat, aux = self.reward(zc, a1h)                       # [B, N]
        rew_loss = F.mse_loss(r_hat, r)
        # Balancing term composed from the router's aux outputs (vendor
        # module untouched): the clone's packaged loss_balancing includes
        # z_loss = log Σ exp(logits), which is UNBOUNDED BELOW — minimizing
        # it drives every logit to −∞ (observed: loss → −54, then log(0)
        # NaNs the gradients). We use Shazeer importance balancing +
        # the standard ST-MoE squared z-loss (bounded at 0) instead.
        gates, logits = aux["gates"], aux["logits"]
        importance = gates.sum(dim=0)
        bal_loss = (self.reward.router.cv_squared(importance).mean()
                    + logits.logsumexp(dim=-1).pow(2).mean())

        total = dyn_loss + rew_loss + self.bal_coef * bal_loss
        parts = {
            "wm_dyn_loss": float(dyn_loss.detach()),
            "wm_rew_mse": float(rew_loss.detach()),
            "wm_bal_loss": float(bal_loss.detach()),
        }
        return total, parts
