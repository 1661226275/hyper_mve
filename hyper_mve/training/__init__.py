"""Pkg-05: v4 training system public API.

Stable imports for Pkg-06/07/08 (spec 08 §6.1):

    from hyper_mve.training import MuZeroTrainer, Worker, EpisodeReplayBuffer
    from hyper_mve.training.curriculum import CurriculumScheduler
    from hyper_mve.training.loss_composition import compose_total_loss
"""
from __future__ import annotations

from hyper_mve.training.curriculum import CurriculumScheduler
from hyper_mve.training.episode_buffer import EpisodeReplayBuffer
from hyper_mve.training.loss_composition import compose_total_loss
from hyper_mve.training.muzero_trainer import MuZeroTrainer
from hyper_mve.training.worker import Worker

__all__ = [
    "MuZeroTrainer",
    "Worker",
    "EpisodeReplayBuffer",
    "CurriculumScheduler",
    "compose_total_loss",
]
