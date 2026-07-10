"""Post-hoc game-theoretic metrics (v5 Pkg-09): NashConv + empirical Price of Anarchy.

The two thesis game-theory metrics, computed OFFLINE on frozen checkpoints
(deliberately outside the frozen per-run EvalReport — BR training is too
expensive to run at every eval):

1. **NashConv / exploitability (per regime)** — the canonical
   closeness-to-equilibrium measure. For each agent *i* and pinned regime
   *g*: freeze the evaluated joint policy π, train an independent DQN
   best-response ``BR_i`` against π_{−i} under the same information
   conditions (own row visible, regime hidden), and report

       ``NashConv(g) = Σ_i max(0, V_i(BR_i, π_{−i}) − V_i(π))``.

   The BR is an *approximate* best response, so the reported value is a
   **lower bound** on true exploitability — state the BR budget alongside it.

2. **Empirical Price of Anarchy (welfare efficiency per regime)** —
   ``Eff(g) = W_phys(π, g) / Ŵ*`` where ``Ŵ*`` is a cooperative-optimum
   reference welfare (physics is regime-independent, so one Ŵ* serves all
   regimes). Ŵ* is itself an estimate (e.g. the welfare of a policy trained
   under the all-coop regime); Eff can exceed 1 if Ŵ* undershoots — report
   Ŵ*'s provenance.

Protocol notes (honesty constraints for the thesis):
  * The frozen policy acts via its **distilled prior** (argmax of the
    prediction net) — the deployable artifact — NOT the MVE planner. Running
    the planner inside BR training would multiply cost by ~mve_samples and
    entangle the metric with planner hyper-parameters.
  * BR sees only its own observation (which carries its own row); the regime
    id stays hidden from BR exactly as from the evaluated policy.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_mve.configs import V4Config
from hyper_mve.envs.relation_commons import RelationCommonsEnv
from hyper_mve.schemas import get_regime_family
from hyper_mve.utils.utils import inverse_scalar_transform


# ---------------------------------------------------------------------------
# Frozen policy (distilled prior of a v5 6-API model)
# ---------------------------------------------------------------------------

class FrozenPriorPolicy:
    """Greedy joint policy from a frozen v5 model's prediction net.

    Maintains the BeliefNet GRU hidden across an episode; conditioning
    follows the worker's row-i-only-for-agent-i discipline.
    """

    def __init__(self, model, cfg: V4Config, device: Optional[torch.device] = None):
        self.model = model.eval()
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        self.N = cfg.env.N
        self.A = cfg.env.A
        self._hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._hidden = self.model.belief_net.init_hidden(1, self.N, device=self.device)

    @torch.no_grad()
    def joint_actions(self, obs: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """obs (N, obs_dim), rows (N, N-1) → greedy joint action (N,) int64."""
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        self._hidden, g_hat = self.model.belief_net.step(obs_t, self._hidden)
        s = self.model.encode(obs_t)
        rows_t = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(self.device)
        actions = np.zeros(self.N, dtype=np.int64)
        for k in range(self.N):
            self.model.set_context_subjective(
                k, rows_t[k].unsqueeze(0), g_hat[:, k],
            )
            logits, _ = self.model.predict(s)
            actions[k] = int(logits.argmax(dim=-1).item())
        return actions

    @torch.no_grad()
    def agent_value(self, obs: np.ndarray, rows: np.ndarray, agent_id: int) -> float:
        """Value estimate for one agent (diagnostic only)."""
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        hidden = self.model.belief_net.init_hidden(1, self.N, device=self.device)
        _, g_hat = self.model.belief_net.step(obs_t, hidden)
        s = self.model.encode(obs_t)
        rows_t = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(self.device)
        self.model.set_context_subjective(agent_id, rows_t[agent_id].unsqueeze(0), g_hat[:, agent_id])
        _, v = self.model.predict(s)
        return float(inverse_scalar_transform(v).item())


# ---------------------------------------------------------------------------
# Small double-DQN best responder
# ---------------------------------------------------------------------------

class _QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class BRConfig:
    """Best-response training budget (report these next to NashConv)."""
    env_steps: int = 20_000
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    target_update_every: int = 500
    train_every: int = 4
    warmup_steps: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    hidden: int = 128


def _epsilon(br: BRConfig, step: int) -> float:
    frac = min(1.0, step / max(1, br.eps_decay_steps))
    return br.eps_start + (br.eps_end - br.eps_start) * frac


def train_best_response(
    cfg: V4Config,
    frozen: FrozenPriorPolicy,
    agent_id: int,
    regime_id: int,
    br: BRConfig,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> _QNet:
    """Train a DQN best response for ``agent_id`` in pinned regime ``regime_id``
    against the frozen policy controlling every other agent."""
    device = device or frozen.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    env = RelationCommonsEnv(cfg.env, seed=seed)
    obs, info = env.reset(seed=seed, options={"g": int(regime_id)})
    frozen.reset()
    obs_dim = obs.shape[-1]
    A = cfg.env.A

    q = _QNet(obs_dim, A, br.hidden).to(device)
    q_target = _QNet(obs_dim, A, br.hidden).to(device)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=br.lr)
    buf: deque = deque(maxlen=br.buffer_size)
    gamma = float(cfg.train.gamma)

    for step in range(br.env_steps):
        joint = frozen.joint_actions(obs, info["rows"])
        obs_i = obs[agent_id].copy()
        if step < br.warmup_steps or rng.random() < _epsilon(br, step):
            a_i = int(rng.integers(A))
        else:
            with torch.no_grad():
                qv = q(torch.from_numpy(obs_i).to(device).unsqueeze(0))
                a_i = int(qv.argmax(dim=-1).item())
        joint[agent_id] = a_i

        next_obs, reward, done, _trunc, next_info = env.step(joint)
        buf.append((obs_i, a_i, float(reward[agent_id]),
                    next_obs[agent_id].copy(), bool(done)))

        obs, info = next_obs, next_info
        if done:
            obs, info = env.reset(options={"g": int(regime_id)})
            frozen.reset()

        if step >= br.warmup_steps and step % br.train_every == 0 and len(buf) >= br.batch_size:
            batch = random.sample(buf, br.batch_size)
            o = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            a = torch.tensor([b[1] for b in batch], device=device)
            r = torch.tensor([b[2] for b in batch], device=device)
            o2 = torch.from_numpy(np.stack([b[3] for b in batch])).to(device)
            d = torch.tensor([float(b[4]) for b in batch], device=device)
            with torch.no_grad():
                a2 = q(o2).argmax(dim=-1)                       # double DQN
                target = r + gamma * (1.0 - d) * q_target(o2).gather(
                    1, a2.unsqueeze(-1)).squeeze(-1)
            pred = q(o).gather(1, a.unsqueeze(-1)).squeeze(-1)
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if step % br.target_update_every == 0:
            q_target.load_state_dict(q.state_dict())

    env.close()
    return q


# ---------------------------------------------------------------------------
# Rollout evaluators
# ---------------------------------------------------------------------------

def _rollout(
    cfg: V4Config,
    frozen: FrozenPriorPolicy,
    regime_id: int,
    episodes: int,
    seed: int,
    br_agent: Optional[int] = None,
    br_q: Optional[_QNet] = None,
) -> tuple[np.ndarray, float]:
    """Deterministic episodes in a pinned regime.

    Returns (mean per-agent subjective returns (N,), mean total physical welfare).
    When ``br_agent``/``br_q`` are given, that agent acts greedily from the BR
    Q-net instead of the frozen prior.
    """
    N = cfg.env.N
    rets = np.zeros((episodes, N), dtype=np.float64)
    phys = np.zeros(episodes, dtype=np.float64)
    for ep in range(episodes):
        env = RelationCommonsEnv(cfg.env, seed=seed + ep)
        obs, info = env.reset(seed=seed + ep, options={"g": int(regime_id)})
        frozen.reset()
        done = False
        while not done:
            joint = frozen.joint_actions(obs, info["rows"])
            if br_q is not None and br_agent is not None:
                with torch.no_grad():
                    qv = br_q(torch.from_numpy(
                        np.asarray(obs[br_agent], dtype=np.float32)
                    ).to(frozen.device).unsqueeze(0))
                joint[br_agent] = int(qv.argmax(dim=-1).item())
            obs, reward, done, _trunc, info = env.step(joint)
            rets[ep] += np.asarray(reward, dtype=np.float64)
            phys[ep] += float(np.asarray(info["harvests"], dtype=np.float64).sum())
        env.close()
    return rets.mean(axis=0), float(phys.mean())


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

@dataclass
class GameMetricsReport:
    """JSON-serialisable game-metrics deliverable for one checkpoint."""
    variant: str
    checkpoint: str
    regime_ids: list[int]
    br_env_steps: int
    eval_episodes: int
    # per-regime
    v_pi: dict[int, list[float]] = field(default_factory=dict)         # V_i(π)
    v_br: dict[int, list[float]] = field(default_factory=dict)         # V_i(BR_i, π_-i)
    exploitability: dict[int, list[float]] = field(default_factory=dict)
    nashconv: dict[int, float] = field(default_factory=dict)
    welfare_physical: dict[int, float] = field(default_factory=dict)
    efficiency: dict[int, float] = field(default_factory=dict)         # empty w/o coop ref
    coop_reference_welfare: Optional[float] = None
    coop_reference_provenance: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": "game-metrics-v1",
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "regime_ids": list(self.regime_ids),
            "br_env_steps": self.br_env_steps,
            "eval_episodes": self.eval_episodes,
            "v_pi": {str(k): v for k, v in self.v_pi.items()},
            "v_br": {str(k): v for k, v in self.v_br.items()},
            "exploitability": {str(k): v for k, v in self.exploitability.items()},
            "nashconv": {str(k): v for k, v in self.nashconv.items()},
            "welfare_physical": {str(k): v for k, v in self.welfare_physical.items()},
            "efficiency": {str(k): v for k, v in self.efficiency.items()},
            "coop_reference_welfare": self.coop_reference_welfare,
            "coop_reference_provenance": self.coop_reference_provenance,
            "note": (
                "BR is an approximate best response (independent double-DQN, "
                "budget br_env_steps) => NashConv values are LOWER BOUNDS on true "
                "exploitability. Frozen policy acts via its distilled prior "
                "(argmax prediction net), not the MVE planner."
            ),
        }


def compute_game_metrics(
    model,
    cfg: V4Config,
    *,
    variant: str = "hyper",
    checkpoint: str = "",
    regime_ids: Optional[list[int]] = None,
    br: Optional[BRConfig] = None,
    eval_episodes: int = 10,
    seed: int = 0,
    coop_reference_welfare: Optional[float] = None,
    coop_reference_provenance: str = "",
    device: Optional[torch.device] = None,
) -> GameMetricsReport:
    """Compute NashConv + welfare (+ efficiency when a coop reference is given)."""
    br = br or BRConfig()
    family = get_regime_family(cfg.env)
    if regime_ids is None:
        regime_ids = list(range(family.size))
    frozen = FrozenPriorPolicy(model, cfg, device=device)
    N = cfg.env.N

    report = GameMetricsReport(
        variant=variant,
        checkpoint=checkpoint,
        regime_ids=list(regime_ids),
        br_env_steps=br.env_steps,
        eval_episodes=eval_episodes,
        coop_reference_welfare=coop_reference_welfare,
        coop_reference_provenance=coop_reference_provenance,
    )

    for g in regime_ids:
        base_returns, welfare = _rollout(cfg, frozen, g, eval_episodes, seed)
        report.v_pi[g] = [float(x) for x in base_returns]
        report.welfare_physical[g] = welfare
        if coop_reference_welfare and coop_reference_welfare > 0:
            report.efficiency[g] = welfare / float(coop_reference_welfare)

        v_br: list[float] = []
        for i in range(N):
            q = train_best_response(cfg, frozen, i, g, br, seed=seed + 1000 * (g + 1) + i)
            br_returns, _ = _rollout(
                cfg, frozen, g, eval_episodes, seed, br_agent=i, br_q=q,
            )
            v_br.append(float(br_returns[i]))
        report.v_br[g] = v_br
        report.exploitability[g] = [
            max(0.0, vb - vp) for vb, vp in zip(v_br, report.v_pi[g])
        ]
        report.nashconv[g] = float(sum(report.exploitability[g]))

    return report


__all__ = [
    "BRConfig",
    "FrozenPriorPolicy",
    "GameMetricsReport",
    "compute_game_metrics",
    "train_best_response",
]
