"""Worker — v4 episode collection (Pkg-05 spec 02, Ch5.6).

Rewrite of the v4.7 worker for the v4 stack:
    - BeliefNet.step online inference (C5-W2) replaces v4.7 set_context_from_history.
    - 7-API per-agent context (set_context_objective once + set_context_subjective
      per agent), no model.update_step (C5-W1: worker is always no_grad inference).
    - emits v4 TimeStepRecord (12 fields) + a parallel c_t scalar sequence.
    - holds a single MVEPlanner instance so the CRN seed persists across episodes
      (review 修订 5).

Self-Info strictness (C11): set_context_subjective only ever gets the RAW (B, 4)
capability vector — never oracle types. Own type is resolved inside the model from
cfg.env.type_assignment, not passed here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.schemas import TimeStepRecord
from hyper_mve.utils.utils import inverse_scalar_transform


class Worker:
    """Collects one episode at a time into v4 TimeStepRecords."""

    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        env,
        planner: Optional[MVEPlanner] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.env = env
        # review 修订 5: hold one planner so CRN rng persists across episodes.
        self.planner = planner if planner is not None else MVEPlanner(cfg)
        self.model.eval()

    @torch.no_grad()
    def collect_episode(
        self,
        epsilon: float = 0.1,
        use_planner: bool = True,
    ) -> tuple[list[TimeStepRecord], torch.Tensor]:
        """Collect a full episode.

        Returns:
            (records, c_t_seq): T TimeStepRecords and the parallel (T,) c_t series
            (the second arg to ``EpisodeReplayBuffer.store_episode``).
        """
        N = self.cfg.env.N
        A = self.cfg.env.A
        device = next(self.model.parameters()).device

        obs, info = self.env.reset()                       # obs (N, obs_dim)
        prev_hidden = self.model.belief_net.init_hidden(1, N, device=device)

        records: list[TimeStepRecord] = []
        c_t_list: list[float] = []

        for t in range(self.cfg.env.T_max):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)

            # --- BeliefNet online inference (C5-W2) ---
            prev_hidden, c_hat, z_hat = self.model.belief_net.step(obs_t, prev_hidden)
            # c_hat (1, N), z_hat (1, N, N-1, 2)

            s = self.model.encode(obs_t)                   # (1, latent_dim)
            c_t_scalar = float(info["c_true"])
            c_t_tensor = torch.tensor([c_t_scalar], dtype=torch.float32, device=device)
            self.model.set_context_objective(c_t_tensor)

            pi_mve = np.zeros((N, A), dtype=np.float32)
            value_estimates = np.zeros(N, dtype=np.float32)
            cap_dict: dict[int, torch.Tensor] = {}
            belief_dict: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

            for k in range(N):
                cap_k = torch.from_numpy(
                    np.asarray(info["caps"][k].to_array(), dtype=np.float32)
                ).unsqueeze(0).to(device)                  # (1, 4) RAW cap (no type leak)
                cap_dict[k] = cap_k
                belief_dict[k] = (c_hat[0, k:k + 1], z_hat[0, k:k + 1])

                self.model.set_context_subjective(k, cap_k, belief_dict[k])
                logits_k, v_k = self.model.predict(s)
                value_estimates[k] = float(inverse_scalar_transform(v_k).item())
                if not use_planner:
                    pi_mve[k] = torch.softmax(logits_k, dim=-1)[0].cpu().numpy()

            if use_planner:
                pi_all = self.planner.sample_mve_plan(
                    self.model, s, cap_dict, belief_dict, c_t_tensor,
                )                                          # (1, N, A)
                pi_mve = pi_all[0].cpu().numpy().astype(np.float32)

            # --- epsilon-greedy action selection (per agent) ---
            joint_action = np.zeros(N, dtype=np.int64)
            for k in range(N):
                if np.random.rand() < epsilon:
                    joint_action[k] = np.random.randint(A)
                else:
                    p = pi_mve[k].astype(np.float64)
                    p = p / p.sum()
                    joint_action[k] = int(np.random.choice(A, p=p))

            next_obs, reward, done, _truncated, next_info = self.env.step(joint_action)

            # --- write TimeStepRecord (C5-W3; Pkg-01 spec 04 12 fields) ---
            cap_record = np.stack(
                [np.asarray(info["caps"][k].to_array(), dtype=np.float32) for k in range(N)],
                axis=0,
            )                                              # (N, 4)
            record = TimeStepRecord(
                o=np.asarray(obs, dtype=np.float32),
                a=joint_action,
                r=np.asarray(reward, dtype=np.float32),
                delta=np.asarray(next_info["deltas"], dtype=np.float32),
                pi_mve=pi_mve.astype(np.float32),
                v=value_estimates,
                tau=np.asarray(info["types"], dtype=np.int8),
                cap=cap_record,
                c_hat=c_hat[0].cpu().numpy().astype(np.float32),
                z_hat=z_hat[0].cpu().numpy().astype(np.float32),
                t=t,
                done=bool(done),
            )
            records.append(record)
            c_t_list.append(c_t_scalar)

            if done:
                break
            obs = next_obs
            info = next_info

        c_t_seq = torch.tensor(c_t_list, dtype=torch.float32)
        return records, c_t_seq
