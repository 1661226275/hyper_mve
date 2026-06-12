"""EpisodeReplayBuffer — v4 TimeStepRecord container (Pkg-05 spec 03, Ch5.6).

Rewrite of the v4.7 ``EpisodeData`` buffer (6 fields) into a container for the
v4 ``TimeStepRecord`` (12 fields, Pkg-01 spec 04). Episodes are stored on CPU as
``dict[str, np.ndarray]`` stacked over the time axis; ``sample_batch`` slices
``(B, K+1, ...)`` windows and ``torch.from_numpy``s them.

review 修订 4: ``store_episode`` takes an explicit parallel ``c_t_seq`` (T,) so the
buffer can return a ``c_t`` field without modifying the TimeStepRecord schema (NG7).

Sampling (D9, inlined): uniform or type-stratified (Ch5.6.5). Stratified keeps at
least ``stratified_min_per_type_frac`` of the batch from each of the alpha-heavy /
beta-heavy episode buckets so the reward head sees both types every step.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.schemas import AgentType, TimeStepRecord


# Fields stacked over the time axis (Pkg-01 spec 04 TimeStepRecord 10 data fields).
_DATA_FIELDS = ("o", "a", "r", "delta", "pi_mve", "v", "tau", "cap", "c_hat", "z_hat")


class EpisodeReplayBuffer:
    """FIFO replay buffer of whole episodes (v4 TimeStepRecord container)."""

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.max_episodes: int = cfg.train.buffer_size
        self.unroll_K: int = cfg.train.unroll_K
        self.n_step: int = cfg.train.n_step
        self.min_buffer_size: int = cfg.train.min_buffer_size

        self._episodes: "deque[dict[str, np.ndarray]]" = deque(maxlen=self.max_episodes)
        self._c_t_seqs: "deque[np.ndarray]" = deque(maxlen=self.max_episodes)
        # [v4-opt 2026-06] parallel per-episode metadata (same FIFO discipline):
        # planner_on=False marks self-distillation pi_mve targets (warmup / debug
        # collection) so the policy loss can mask them; collected_at_step feeds the
        # diag/target_age_steps staleness probe.
        self._planner_on: "deque[bool]" = deque(maxlen=self.max_episodes)
        self._collected_at: "deque[int]" = deque(maxlen=self.max_episodes)

        self.stratified: bool = cfg.train.stratified_sampling
        self.stratified_min_frac: float = cfg.train.stratified_min_per_type_frac

    # ------------------------------------------------------------- store

    def store_episode(
        self,
        records: list[TimeStepRecord],
        c_t_seq: torch.Tensor,
        planner_on: bool = True,
        collected_at_step: int = 0,
    ) -> None:
        """Store one episode.

        Args:
            records: T ``TimeStepRecord`` (Pkg-01 spec 04).
            c_t_seq: (T,) float32 — parallel c_t scalar series (worker collects it
                from ``env.info['c_true']`` per step).
            planner_on: False when the episode's pi_mve came from the model's own
                prior (warmup / --no_collect_planner) — those targets are
                self-distillation and the policy loss masks them [v4-opt 2026-06].
            collected_at_step: global_step at collection time (staleness probe).
        """
        T = len(records)
        assert T == len(c_t_seq), (
            f"records len {T} != c_t_seq len {len(c_t_seq)}"
        )
        assert T > self.unroll_K + self.n_step, (
            f"episode too short: T={T} <= unroll_K+n_step={self.unroll_K + self.n_step}"
        )

        episode = {
            "o":      np.stack([r.o for r in records], axis=0),       # (T, N, obs_dim)
            "a":      np.stack([r.a for r in records], axis=0),       # (T, N)
            "r":      np.stack([r.r for r in records], axis=0),       # (T, N)
            "delta":  np.stack([r.delta for r in records], axis=0),   # (T, N)
            "pi_mve": np.stack([r.pi_mve for r in records], axis=0),  # (T, N, A)
            "v":      np.stack([r.v for r in records], axis=0),       # (T, N)
            "tau":    np.stack([r.tau for r in records], axis=0),     # (T, N) int8
            "cap":    np.stack([r.cap for r in records], axis=0),     # (T, N, 4)
            "c_hat":  np.stack([r.c_hat for r in records], axis=0),   # (T, N)
            "z_hat":  np.stack([r.z_hat for r in records], axis=0),   # (T, N, N-1, 2)
            "t":      np.array([r.t for r in records], dtype=np.int32),     # (T,)
            "done":   np.array([r.done for r in records], dtype=np.bool_),  # (T,)
        }
        self._episodes.append(episode)
        self._c_t_seqs.append(np.asarray(c_t_seq.detach().cpu().numpy(), dtype=np.float32))
        self._planner_on.append(bool(planner_on))
        self._collected_at.append(int(collected_at_step))

    # ------------------------------------------------------------ sample

    def sample_batch(self, batch_size: int, unroll_K: Optional[int] = None) -> dict[str, torch.Tensor]:
        """Sample a (B, K+1, ...) batch dict (see module / spec 03 §2.2)."""
        K = self.unroll_K if unroll_K is None else unroll_K
        assert len(self._episodes) > 0, "Buffer is empty; collect episodes first."

        if self.stratified:
            ep_indices, start_indices = self._stratified_sample(batch_size, K)
        else:
            ep_indices, start_indices = self._uniform_sample(batch_size, K)

        return self._slice_batch(ep_indices, start_indices, K)

    def __len__(self) -> int:
        return len(self._episodes)

    # ----------------------------------------------------------- helpers

    def _max_start(self, ep_idx: int, K: int) -> int:
        """Largest valid window start so that ``[start, start+K+1)`` fits."""
        T_ep = len(self._episodes[ep_idx]["t"])
        return max(0, T_ep - (K + 1))

    def _uniform_sample(self, B: int, K: int) -> tuple[list[int], list[int]]:
        ep_indices: list[int] = []
        start_indices: list[int] = []
        n_ep = len(self._episodes)
        for _ in range(B):
            ep_idx = int(np.random.randint(n_ep))
            start = int(np.random.randint(0, self._max_start(ep_idx, K) + 1))
            ep_indices.append(ep_idx)
            start_indices.append(start)
        return ep_indices, start_indices

    def _stratified_sample(self, B: int, K: int) -> tuple[list[int], list[int]]:
        """Type-stratified sampling (D9, C5-B2).

        Bucket episodes by the first-step type assignment (fixed within an episode,
        Pkg-02 spec 03): alpha-heavy (>=50% ALPHA) vs beta-heavy. Draw at least
        ``B * stratified_min_per_type_frac`` from each non-empty bucket, fill the
        rest uniformly, then shuffle to avoid intra-batch ordering bias.
        """
        alpha_heavy: list[int] = []
        beta_heavy: list[int] = []
        for i, ep in enumerate(self._episodes):
            taus = ep["tau"][0]  # (N,) first-step types
            alpha_frac = (taus == int(AgentType.ALPHA)).sum() / len(taus)
            (alpha_heavy if alpha_frac >= 0.5 else beta_heavy).append(i)

        ep_indices: list[int] = []
        start_indices: list[int] = []
        min_per_type = int(B * self.stratified_min_frac)

        def _draw_from(bucket: list[int]) -> None:
            ep_idx = int(np.random.choice(bucket))
            start = int(np.random.randint(0, self._max_start(ep_idx, K) + 1))
            ep_indices.append(ep_idx)
            start_indices.append(start)

        for _ in range(min_per_type):
            if alpha_heavy:
                _draw_from(alpha_heavy)
            if beta_heavy:
                _draw_from(beta_heavy)

        remaining = B - len(ep_indices)
        if remaining > 0:
            ep_u, start_u = self._uniform_sample(remaining, K)
            ep_indices.extend(ep_u)
            start_indices.extend(start_u)
        elif remaining < 0:
            # Both buckets present and 2*min_per_type > B: trim to B.
            ep_indices = ep_indices[:B]
            start_indices = start_indices[:B]

        perm = np.random.permutation(len(ep_indices))
        ep_indices = [ep_indices[i] for i in perm]
        start_indices = [start_indices[i] for i in perm]
        return ep_indices, start_indices

    def _slice_batch(self, ep_indices: list[int], start_indices: list[int], K: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        for key in (*_DATA_FIELDS, "t", "done"):
            slices = [
                self._episodes[ep_idx][key][start:start + K + 1]
                for ep_idx, start in zip(ep_indices, start_indices)
            ]
            out[key] = torch.from_numpy(np.stack(slices, axis=0))  # (B, K+1, ...)

        # Rename to the trainer's expected keys.
        out["obs"] = out.pop("o")
        out["actions"] = out.pop("a")
        out["rewards"] = out.pop("r")
        out["dones"] = out.pop("done")

        # c_t: parallel slice from _c_t_seqs (review 修订 4).
        c_t_slices = [
            self._c_t_seqs[ep_idx][start:start + K + 1]
            for ep_idx, start in zip(ep_indices, start_indices)
        ]
        out["c_t"] = torch.from_numpy(np.stack(c_t_slices, axis=0))  # (B, K+1)

        # [v4-opt 2026-06] per-episode metadata: policy-loss mask + staleness probe.
        out["planner_on"] = torch.tensor(
            [self._planner_on[ep_idx] for ep_idx in ep_indices], dtype=torch.bool,
        )                                                            # (B,)
        out["collected_at_step"] = torch.tensor(
            [self._collected_at[ep_idx] for ep_idx in ep_indices], dtype=torch.long,
        )                                                            # (B,)

        return out
