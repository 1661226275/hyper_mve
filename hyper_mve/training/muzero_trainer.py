"""MuZeroTrainer — v4 single train_step (Pkg-05 spec 01 + spec 07, Ch5.6).

Q2/D1: the v4.7 ``train_step`` + ``train_step_infer`` pair is merged into one
``train_step`` — BeliefNet (Pkg-03) replaces the v4.7 GRU rule inference, so
"Oracle vs Infer" is no longer a model-class distinction but a curriculum stage.

Per train_step (v5 order, enforced by compose_total_loss):
    model.update_step(step) + target.update_step(step)      (C5-T1)
    -> compose_total_loss: for k in range(N):
       set_context_subjective(agent, row, g_main)            (C5-T3, v5)
    -> total.backward() -> clip -> optimizer.step()
    -> lr_scheduler.step() -> EMA target update

D6: EMA + LR scheduler are inlined helpers (spec 07). D8: v4 checkpoints carry a
``version`` field and refuse to load v4.7 checkpoints.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.training.curriculum import CurriculumScheduler
from hyper_mve.training.loss_composition import compose_total_loss


_TRAIN_REQUIRED = [
    "max_train_steps", "batch_size", "buffer_size", "min_buffer_size",
    "episodes_per_iter", "train_steps_per_iter",
    "unroll_K", "n_step", "gamma",
    "lr", "lr_min", "adam_eps", "grad_clip",
    "lr_schedule", "lr_warmup_steps",
    "w_policy", "w_value", "w_reward", "w_consist", "w_belief",
    "curriculum_stage_1_end_frac", "curriculum_stage_2_end_frac",
    "ema_tau", "stratified_sampling",
]
_ENV_REQUIRED = ["N", "A"]
_MODEL_REQUIRED = ["latent_dim"]


class MuZeroTrainer:
    """v4 unified MuZero trainer (single train_step)."""

    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        scheduler: Optional[CurriculumScheduler] = None,
        projector: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self._validate_cfg_fields()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Curriculum scheduler (review 修订 1: replaceable instance, D4).
        self.scheduler = scheduler or CurriculumScheduler(cfg)

        # BYOL projector (optional; None -> consistency loss skipped).
        self.projector = projector.to(self.device) if projector is not None else None

        # EMA target network (v4.4): frozen deep copy used for n-step bootstrap.
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.target_model.requires_grad_(False)

        params = list(self.model.parameters())
        if self.projector is not None:
            params += list(self.projector.parameters())
        self._clip_params = params
        self.optimizer = torch.optim.Adam(params, lr=cfg.train.lr, eps=cfg.train.adam_eps)
        self.lr_scheduler = self._build_scheduler()

        self.global_step: int = 0

    def _validate_cfg_fields(self) -> None:
        for f in _TRAIN_REQUIRED:
            assert hasattr(self.cfg.train, f), f"cfg.train missing field: {f}"
        for f in _ENV_REQUIRED:
            assert hasattr(self.cfg.env, f), f"cfg.env missing field: {f}"
        for f in _MODEL_REQUIRED:
            assert hasattr(self.cfg.model, f), f"cfg.model missing field: {f}"

    # ------------------------------------------------------------ train_step

    def train_step(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]:
        """One v4 train step; returns a float dict for logging."""
        self.global_step = global_step
        self.model.train()

        batch = {k: v.to(self.device) for k, v in batch.items()}

        # C5-T1: update step (gating threshold) on both online + target.
        self.model.update_step(global_step)
        self.target_model.update_step(global_step)

        # C5-T2/T3 + double-path loss assembly (spec 05).
        losses = compose_total_loss(self.model, batch, self, global_step, self.cfg)

        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self._clip_params, self.cfg.train.grad_clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        self._update_ema_target()

        out = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in losses.items()}
        out["lr"] = self.lr_scheduler.get_last_lr()[0]
        # [v4-opt 2026-06] target staleness: mean age (train steps) of the sampled
        # episodes' stored pi_mve targets (buffer=5000 episodes can hold ~5000 steps
        # of history; stale CE rises mechanically once the model moves fast).
        if "collected_at_step" in batch:
            out["diag_target_age_steps"] = float(global_step) - batch["collected_at_step"].float().mean().item()
        return out

    # ------------------------------------------------------ n-step return

    def compute_n_step_return(
        self,
        rewards: torch.Tensor,   # (B, K+1, N)
        v_target: torch.Tensor,  # (B, K+1, N) target V at each obs step (original scale)
        dones: torch.Tensor,     # (B, K+1) bool
        n: int,
    ) -> torch.Tensor:           # (B, K+1, N) original scale
        """Truncated n-step return within the unroll window (v4.4 EMA bootstrap).

        z_t = Σ_{j=0}^{n'-1} γ^j r_{t+j}  + γ^{n'} V_target(s_{t+n'})
        with n' = min(n, K - t) so the bootstrap index stays inside the window;
        a ``done`` at step t+j zeros all subsequent rewards and the bootstrap.
        """
        B, T, N = rewards.shape
        gamma = self.cfg.train.gamma
        z = torch.zeros_like(rewards)

        for t in range(T):
            n_eff = min(n, T - 1 - t)
            ret = torch.zeros(B, N, device=rewards.device)
            discount = 1.0
            not_done = torch.ones(B, 1, device=rewards.device)
            for j in range(n_eff):
                idx = t + j
                ret = ret + discount * rewards[:, idx] * not_done
                discount *= gamma
                not_done = not_done * (~dones[:, idx]).float().unsqueeze(-1)
            boot_idx = t + n_eff
            ret = ret + discount * v_target[:, boot_idx] * not_done
            z[:, t] = ret
        return z

    # --------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str) -> None:
        """Save a v4 checkpoint (D8; carries ``version='v4'``)."""
        torch.save({
            "version": "v4",
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "target_state": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "projector_state": (self.projector.state_dict() if self.projector is not None else None),
            "cfg": self.cfg.to_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load a v4 checkpoint, returning ``global_step`` (D8: refuse v4.7)."""
        ckpt = torch.load(path, map_location=self.device)
        version = ckpt.get("version", "v4.7")
        if version != "v4":
            raise RuntimeError(
                f"Checkpoint version mismatch: expected 'v4', got '{version}'. "
                "v4 与 v4.7 不兼容 (Pkg-05 D8). "
                "若需 reproduce v4.7 实验: git checkout v4.7-final + _legacy_v4_7/scripts/."
            )
        self.global_step = ckpt["global_step"]
        self.model.load_state_dict(ckpt["model_state"])
        self.target_model.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        if self.projector is not None and ckpt.get("projector_state") is not None:
            self.projector.load_state_dict(ckpt["projector_state"])
        return self.global_step

    # --------------------------------------------------- internal helpers

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """warmup_cosine / cosine / multistep (spec 07 §2.2)."""
        t = self.cfg.train
        if t.lr_schedule == "warmup_cosine":
            min_ratio = t.lr_min / t.lr

            def lr_lambda(step: int) -> float:
                if step < t.lr_warmup_steps:
                    return step / max(1, t.lr_warmup_steps)
                progress = (step - t.lr_warmup_steps) / max(1, t.max_train_steps - t.lr_warmup_steps)
                progress = min(progress, 1.0)
                return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if t.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t.max_train_steps, eta_min=t.lr_min,
            )

        if t.lr_schedule == "multistep":
            milestones = getattr(t, "lr_milestones", [60000, 120000])
            gamma = getattr(t, "lr_gamma", 0.5)
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma,
            )

        raise ValueError(f"Unknown lr_schedule: {t.lr_schedule!r}")

    def _update_ema_target(self) -> None:
        """EMA: target <- tau * target + (1 - tau) * online (v4.4, tau=0.99)."""
        with torch.no_grad():
            tau = self.cfg.train.ema_tau
            for p_o, p_t in zip(self.model.parameters(), self.target_model.parameters()):
                p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
