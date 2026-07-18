"""GreedyModelPlanner — greedy-sampling planning through the MoE world model
(the user-locked replacement for upstream m3w's MPPI, which is Box-only).

At each real step: sample K joint-action candidates from the SAC policies
(candidate 0 = the joint argmax; under ``explore=True`` the tail includes
uniform-random joints), roll each candidate H steps through the world model
(follow-on actions = policy argmax at imagined latents), accumulate
discounted PER-AGENT predicted rewards plus a soft-value bootstrap, and let
each agent GREEDILY execute its own action from its own best-scoring joint
candidate (per-agent greedy — general-sum semantics, no joint argmax).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GreedyModelPlanner:
    def __init__(
        self,
        wm,
        sac,
        *,
        n_candidates: int = 16,
        horizon: int = 3,
        gamma: float = 0.99,
        n_random: int = 2,
    ) -> None:
        self.wm = wm
        self.sac = sac
        self.n_candidates = int(n_candidates)
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.n_random = int(n_random)

    @torch.no_grad()
    def plan(
        self,
        obs: np.ndarray,
        g: int,
        *,
        explore: bool,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """obs [N, obs_dim] (numpy), g = the GIVEN regime ID → actions [N]."""
        device = next(self.wm.parameters()).device
        N, A = self.wm.n_agents, self.wm.n_actions
        K = self.n_candidates

        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        g_t = torch.full((1,), int(g), dtype=torch.long, device=device)
        zc0 = self.wm.encode(obs_t, g_t)                       # [1, N, d_zc]

        logits0 = self.sac.action_logits(zc0)                  # [1, N, A]
        probs0 = F.softmax(logits0, dim=-1).reshape(N, A)
        cand = torch.multinomial(
            probs0, K, replacement=True, generator=generator).T  # [K, N]
        cand[0] = logits0.argmax(dim=-1).reshape(N)            # joint argmax
        if explore and self.n_random > 0 and K > 1 + self.n_random:
            rnd = torch.randint(
                0, A, (self.n_random, N), device=device, generator=generator)
            cand[-self.n_random:] = rnd

        z = zc0.expand(K, N, self.wm.d_zc).contiguous()
        a = cand
        score = torch.zeros(K, N, device=device)
        for h in range(self.horizon):
            score += (self.gamma ** h) * self.wm.predict_rewards(z, a)
            z = self.wm.predict_next(z, a)
            a = self.sac.action_logits(z).argmax(dim=-1)       # follow-on
        score += (self.gamma ** self.horizon) * self.sac.expected_value(z)

        best_k = score.argmax(dim=0)                           # [N]
        actions = cand[best_k, torch.arange(N, device=device)]
        return actions.cpu().numpy().astype(np.int64)
