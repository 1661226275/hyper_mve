"""PerAgentDiscreteSAC — HASAC-derived per-agent discrete SAC on the
regime-conditioned latent.

Adaptation of HASAC (HARL) to the M3W-adapted spec: per-agent actors and
twin Q critics over the conditioned latent ``zc_i`` (which already carries
the given regime embedding), PER-AGENT rewards (general-sum), discrete
actions via HASAC's gumbel-softmax straight-through mechanism in the actor
update. The critic target uses the exact expected form over the discrete
action set (the closed form of HASAC's sampled soft-value estimate).
Critics are decentralized (per-agent, own latent) — centralization lives in
the MoE world model; disclosed in the runner docstring.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in: int, d_out: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, d_out),
    )


class PerAgentDiscreteSAC(nn.Module):
    def __init__(
        self,
        *,
        n_agents: int,
        d_in: int,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.05,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gumbel_tau = float(gumbel_tau)

        self.actors = nn.ModuleList(
            [_mlp(d_in, n_actions) for _ in range(self.n_agents)])
        self.q1s = nn.ModuleList(
            [_mlp(d_in, n_actions) for _ in range(self.n_agents)])
        self.q2s = nn.ModuleList(
            [_mlp(d_in, n_actions) for _ in range(self.n_agents)])
        self.q1_targets = nn.ModuleList(
            [_mlp(d_in, n_actions) for _ in range(self.n_agents)])
        self.q2_targets = nn.ModuleList(
            [_mlp(d_in, n_actions) for _ in range(self.n_agents)])
        for src, dst in ((self.q1s, self.q1_targets), (self.q2s, self.q2_targets)):
            for m_src, m_dst in zip(src, dst):
                m_dst.load_state_dict(m_src.state_dict())
                for p in m_dst.parameters():
                    p.requires_grad_(False)

    # ---------------------------------------------------------------- infer
    def action_logits(self, zc: torch.Tensor) -> torch.Tensor:
        """zc [B, N, d_in] → logits [B, N, A]."""
        return torch.stack(
            [self.actors[i](zc[:, i]) for i in range(self.n_agents)], dim=1)

    @torch.no_grad()
    def act(
        self, zc: torch.Tensor, *, greedy: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """zc [B, N, d_in] → actions [B, N] long."""
        logits = self.action_logits(zc)
        if greedy:
            return logits.argmax(dim=-1)
        B, N, A = logits.shape
        probs = F.softmax(logits, dim=-1).reshape(B * N, A)
        a = torch.multinomial(probs, 1, generator=generator)
        return a.reshape(B, N)

    @torch.no_grad()
    def expected_value(self, zc: torch.Tensor) -> torch.Tensor:
        """Soft state value per agent: V_i(zc) [B, N] (planner bootstrap)."""
        logits = self.action_logits(zc)
        pi = F.softmax(logits, dim=-1)
        logpi = F.log_softmax(logits, dim=-1)
        q_min = torch.stack([
            torch.min(self.q1_targets[i](zc[:, i]), self.q2_targets[i](zc[:, i]))
            for i in range(self.n_agents)
        ], dim=1)                                              # [B, N, A]
        return (pi * (q_min - self.alpha * logpi)).sum(dim=-1)

    # ---------------------------------------------------------------- learn
    def update(
        self,
        zc: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
        zc_next: torch.Tensor, done: torch.Tensor,
        q_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """One SAC step on real transitions (latents detached from the WM).

        zc/zc_next [B, N, d_in]; a [B, N] long; r [B, N]; done [B] (terminal
        only — time-limit truncations bootstrap).
        """
        B = zc.shape[0]
        q_losses, actor_losses = [], []

        with torch.no_grad():
            next_logits = self.action_logits(zc_next)
            next_pi = F.softmax(next_logits, dim=-1)
            next_logpi = F.log_softmax(next_logits, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                q_next = torch.min(self.q1_targets[i](zc_next[:, i]),
                                   self.q2_targets[i](zc_next[:, i]))
                v_next = (next_pi[:, i] * (q_next - self.alpha * next_logpi[:, i])
                          ).sum(dim=-1)                        # [B]
                y = r[:, i] + self.gamma * (1.0 - done) * v_next

            a_i = a[:, i].long().view(B, 1)
            q1_a = self.q1s[i](zc[:, i]).gather(1, a_i).squeeze(1)
            q2_a = self.q2s[i](zc[:, i]).gather(1, a_i).squeeze(1)
            q_losses.append(F.mse_loss(q1_a, y) + F.mse_loss(q2_a, y))

        q_loss = torch.stack(q_losses).sum()
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        for i in range(self.n_agents):
            logits = self.actors[i](zc[:, i])
            # HASAC's discrete mechanism: straight-through gumbel-softmax
            a_soft = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
            logpi = F.log_softmax(logits, dim=-1)
            q_min = torch.min(self.q1s[i](zc[:, i]), self.q2s[i](zc[:, i]))
            q_a = (q_min.detach() * a_soft).sum(dim=-1)
            logpi_a = (logpi * a_soft).sum(dim=-1)
            actor_losses.append((self.alpha * logpi_a - q_a).mean())

        actor_loss = torch.stack(actor_losses).sum()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        with torch.no_grad():
            for src, dst in ((self.q1s, self.q1_targets),
                             (self.q2s, self.q2_targets)):
                for m_src, m_dst in zip(src, dst):
                    for p_src, p_dst in zip(m_src.parameters(),
                                            m_dst.parameters()):
                        p_dst.lerp_(p_src, self.tau)

        return {
            "sac_q_loss": float(q_loss.detach()) / self.n_agents,
            "sac_actor_loss": float(actor_loss.detach()) / self.n_agents,
        }
