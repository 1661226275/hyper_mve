"""
Data collection worker for Hyper-MuZero.

Runs episodes in the environment using:
    1. Model's learned policy (PredictionNet) for each agent
    2. Epsilon-greedy exploration
    3. MVE Planner for search policy targets

Produces EpisodeData for the replay buffer.
"""
import copy
import numpy as np
import torch

from training.episode_buffer import EpisodeData
from planning.mve_planner import sample_mve_plan, _is_hyper_model
from utils.utils import actions_to_one_hot


def _is_infer_model(model):
    """Check if model is an Infer-HyperMuZero model (has set_context_from_history)."""
    return hasattr(model, 'set_context_from_history')


class Worker:
    """
    Collects episodes using the current model.

    Each step:
        1. Encode joint obs -> latent state
        2. Run MVE planner -> search policy (pi_mve)
        3. Select action via epsilon-greedy over pi_mve
        4. Step environment
        5. Store transition data
    """

    def __init__(self, env, model, cfg, device):
        """
        Args:
            env:    NonStationaryMultiAgentEnv (discrete)
            model:  BaselineModel (or HyperMuZeroModel)
            cfg:    config
            device: torch device
        """
        self.env = env
        self.model = model
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def collect_episode(self, epsilon=0.1, use_planner=True):
        """
        Run one full episode and return EpisodeData.

        Args:
            epsilon:      exploration rate for epsilon-greedy
            use_planner:  if True, use MVE planner for search policy;
                          if False, use raw model policy (faster, for early warmup)

        Returns:
            EpisodeData with all fields populated
        """
        env = self.env
        model = self.model
        cfg = self.cfg

        obs_n, rule = env.reset()

        # Pre-allocate storage
        T = cfg.episode_limit
        N = cfg.num_agents
        A = cfg.num_actions

        obs_list = []       # list of joint_obs arrays
        action_list = []    # list of (N,) int arrays
        reward_list = []    # list of (N,) float arrays
        policy_list = []    # list of (N, A) float arrays
        done_list = []      # list of bool

        # Determine model type once (invariant within episode)
        is_hyper = _is_hyper_model(model)
        is_infer = _is_infer_model(model)

        for t in range(T):
            # Build joint observation
            joint_obs = np.concatenate(obs_n)  # (joint_obs_dim,)
            obs_list.append(joint_obs)

            # Encode to latent state
            joint_obs_t = torch.tensor(joint_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            s = model.encode(joint_obs_t)  # (1, latent_dim)

            # For Infer model: build history context from past steps
            if is_infer and t > 0:
                W = min(t, cfg.trajectory_window)
                h_obs = torch.tensor(
                    np.array(obs_list[-W:]), dtype=torch.float32
                ).unsqueeze(0).to(self.device)  # (1, W, obs_dim)
                # One-hot encode past actions
                h_act_np = np.zeros((W, N * A), dtype=np.float32)
                for j in range(W):
                    act_j = action_list[t - W + j]
                    for a_idx in range(N):
                        h_act_np[j, a_idx * A + act_j[a_idx]] = 1.0
                h_act = torch.tensor(h_act_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                h_rew_np = np.array([r.mean() for r in reward_list[-W:]], dtype=np.float32)
                h_rew = torch.tensor(h_rew_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                infer_history = (h_obs, h_act, h_rew)
            else:
                infer_history = None

            # Prepare rule tensor for Oracle HyperMuZero
            if is_hyper and not is_infer:
                rule_t = torch.tensor([rule], dtype=torch.float32, device=self.device)
            else:
                rule_t = None

            if use_planner:
                # MVE Planner -> search policy: (1, N, A)
                if is_hyper or is_infer:
                    # v4 migration deferred to Pkg-05 (Q2 折中); see Pkg-04 spec 08 §3.3.
                    # v4.7 sites: set_context_from_history [orig L125] /
                    #             set_context_default [orig L128] /
                    #             sample_mve_plan(..., rule=...) [orig L130/L132].
                    # v4 worker: BeliefNet.step online -> set_context_objective +
                    # set_context_subjective; planner called as
                    # sample_mve_plan(model, s, cfg, c_t=, cap=, belief=).
                    raise NotImplementedError(
                        "v4 hyper/infer worker collection deferred to Pkg-05 (spec 08 §3.3)."
                    )
                else:
                    # Baseline: planner needs no per-agent context.
                    pi_mve = sample_mve_plan(model, s, cfg)
                pi_mve_np = pi_mve.squeeze(0).cpu().numpy()  # (N, A)
            else:
                # Use raw model policy for each agent
                pi_mve_np = np.zeros((N, A), dtype=np.float32)
                for i in range(N):
                    id_i = torch.tensor([i], dtype=torch.long, device=self.device)
                    if is_hyper or is_infer:
                        # v4 migration deferred to Pkg-05 (Q2 折中); see Pkg-04 spec 08 §3.3.
                        # v4.7 sites: set_context_from_history [orig L143] /
                        #             set_context_default [orig L145] /
                        #             set_context(rule_t, id_i) [orig L149].
                        # v4: set_context_objective + set_context_subjective per agent.
                        raise NotImplementedError(
                            "v4 hyper/infer worker raw-policy deferred to Pkg-05 (spec 08 §3.3)."
                        )
                    else:
                        # Baseline model: pass id_emb explicitly
                        id_emb_i = model.get_id_emb(id_i)
                        logits_i, _ = model.predict(s, id_emb_i)  # (1, A)
                    probs_i = torch.softmax(logits_i, dim=-1).squeeze(0).cpu().numpy()
                    pi_mve_np[i] = probs_i

            policy_list.append(pi_mve_np)
            
            # Epsilon-greedy action selection per agent
            actions = np.zeros(N, dtype=np.int64)
            for i in range(N):
                if np.random.random() < epsilon:
                    actions[i] = np.random.randint(0, A)
                else:
                    # Greedy from search policy
                    actions[i] = np.argmax(pi_mve_np[i])

            action_list.append(actions)

            # Step environment (MUST deepcopy to avoid MPE modifying actions)
            obs_next_n, reward_n, done_n, info_n = env.step(copy.deepcopy(actions.tolist()))

            reward_list.append(np.array(reward_n, dtype=np.float32))
            done_list.append(any(done_n))

            obs_n = obs_next_n

            if any(done_n):
                break

        # Append final observation for consistency loss target
        final_obs = np.concatenate(obs_n)
        obs_list.append(final_obs)

        # Convert to arrays
        actual_length = len(action_list)
        obs_arr = np.array(obs_list, dtype=np.float32)          # (T+1, joint_obs_dim)
        actions_arr = np.array(action_list, dtype=np.int64)      # (T, N)
        rewards_arr = np.array(reward_list, dtype=np.float32)    # (T, N)
        policies_arr = np.array(policy_list, dtype=np.float32)   # (T, N, A)
        dones_arr = np.array(done_list, dtype=bool)              # (T,)

        episode = EpisodeData(
            obs=obs_arr,
            actions=actions_arr,
            rewards=rewards_arr,
            search_policies=policies_arr,
            rule=rule,
            dones=dones_arr,
            length=actual_length,
        )

        return episode

    def collect_episodes(self, num_episodes, epsilon=0.1, use_planner=True):
        """
        Collect multiple episodes.

        Args:
            num_episodes: number of episodes to collect
            epsilon:      exploration rate
            use_planner:  whether to use MVE planner

        Returns:
            list of EpisodeData
        """
        episodes = []
        for _ in range(num_episodes):
            ep = self.collect_episode(epsilon=epsilon, use_planner=use_planner)
            episodes.append(ep)
        return episodes