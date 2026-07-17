"""Vendored MAMBA core (pkg-07 spec 06 §4.4 sourcing amendment, 2026-07-10).

Upstream: https://github.com/jbr-ai-labs/mamba @ 2c97258 ("Fix memory leak
for EnvCurriculum.py") — Egorov & Shpilman, "Scalable Multi-Agent Model-Based
Reinforcement Learning", AAMAS 2022. Local clone: ``hyper_mve/baselines/mamba/``.

Port scope (py3.7/torch1.7/ray/wandb/flatland → py3.11/torch2.9, in-process):

  * ``config.py``            ← configs/dreamer/DreamerAgentConfig.py (+ learner /
                               controller config constants folded in)
  * ``transformer_layers.py``← networks/transformer/layers.py
  * ``nets_utils.py``        ← networks/dreamer/utils.py
  * ``dense.py``             ← networks/dreamer/dense.py
  * ``vae.py``               ← networks/dreamer/vae.py
  * ``action.py``            ← networks/dreamer/action.py
  * ``critic.py``            ← networks/dreamer/critic.py
  * ``rnns.py``              ← networks/dreamer/rnns.py
  * ``model.py``             ← agent/models/DreamerModel.py
  * ``memory.py``            ← agent/memory/DreamerMemory.py
  * ``optim_utils.py``       ← agent/optim/utils.py (+ agent/utils/params.py)
  * ``loss.py``              ← agent/optim/loss.py
  * ``learner.py``           ← agent/learners/DreamerLearner.py

Edit classes (every edit carries a ``[PORT]`` comment at the site):
  1. absolute → relative imports;
  2. wandb removed — ``loss.LOGGER`` / learner metrics-callback hook instead;
  3. ray worker/server/runner NOT vendored — episode collection lives in
     ``hyper_mve.comparison.mamba._RealMAMBA`` (in-process, PettingZoo);
  4. flatland / ``environments.Env`` removed — STARCRAFT-branch semantics kept
     everywhere (av_action head exists and is trained on all-ones availability;
     ``critic=self.critic`` in the agent update; ``advantage()`` normalisation;
     FLATLAND-only ``old_critic``/``TARGET_UPDATE`` machinery deleted);
  5. torch 1.7 → 2.9: ``orthogonal_init``'s ``torch.svd`` path replaced with
     ``torch.nn.init.orthogonal_`` (same intent, maintained API).

Upstream behaviour deliberately KEPT (disclosed in the thesis/report):
imagination-phase reward is averaged across agents (``loss.critic_rollout``:
``imag_reward.mean(-2, keepdim=True)``) — MAMBA optimises the team-mean return
(cooperative assumption). The world-model reward head still trains on
per-agent rewards.
"""
