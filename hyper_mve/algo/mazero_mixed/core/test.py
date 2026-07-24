import os
import ray
import time
import torch
from typing import List

import numpy as np
from torch.cuda.amp import autocast as autocast
from gymnasium.utils import seeding

from core.config import BaseConfig, Game
from core.model import BaseNet
from core.mcts import SampledMCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


def predict_rewards_from_model(model, episode, device):
    """fidelity-v1 hook body (one-step per-agent reward predictions, (T, N)):
    belief-GRU rollout over the real observation prefix, then the learned
    reward head scores the realized joint action via recurrent_inference.

    Factored out so both the post-hoc MAZeroMixedRunner.predict_rewards
    (hyper_mve/algo/runner.py) and the periodic in-training fidelity probe
    (train_sync_serial below) share one implementation instead of drifting.
    None if the model has no belief_net (plain-MAZero ablation arms are N/A,
    same as the post-hoc path).
    """
    if not hasattr(model, "belief_net"):
        return None
    was_training = model.training
    model.eval()
    obs_seq = np.asarray(episode["obs"], dtype=np.float32)   # (T+1, N, D)
    actions = np.asarray(episode["actions"], dtype=np.int64)  # (T, N)
    T, N = actions.shape
    preds = np.zeros((T, N), dtype=np.float64)
    try:
        with torch.no_grad():
            hidden = model.belief_net.init_hidden(1, N, device=device)
            for t in range(T):
                obs_t = torch.from_numpy(obs_seq[t]).unsqueeze(0).to(device)
                hidden, g_hat = model.belief_net.step(obs_t, hidden)
                model.set_belief(g_hat)
                out = model.initial_inference(obs_t)
                a_t = torch.from_numpy(actions[t]).reshape(1, N).to(device)
                out2 = model.recurrent_inference(
                    torch.as_tensor(out.hidden_state).to(device), a_t)
                preds[t] = np.asarray(out2.reward).reshape(N)
    finally:
        if was_training:
            model.train()
    return preds


def test(
    config: BaseConfig,
    model: BaseNet,
    counter: int,
    test_episodes: int,
    envs: List[Game] = None,
    np_random: np.random.RandomState = None,
    save_video: bool = False,
    pin_g: int = None,
    sum_agents: bool = False,
    verbose: bool = True,
    device=None,
):
    """evaluation test
    Parameters
    ----------
    model: any
        models for evaluation
    counter: int
        current training step counter
    test_episodes: int
        number of test episodes
        True -> use tqdm bars
    pin_g: int | sequence | None
        None keeps the env's own regime sampling; a scalar forces every temp env
        into that regime; a per-env sequence forces env i into ``pin_g[i]`` (one
        mixed-regime batch search). Used by the per-regime periodic reward probe.
        Eval stays oracle-free (belief inferred from obs, options={"g": g} only
        pins the environment's regime).
    sum_agents: bool
        per-episode score sums the per-agent reward vector (team return, matches
        the external baselines' probe) instead of averaging over agents. The
        summed per-episode scores are returned in ``test_logs['scores']``.
    verbose: bool
        gate the stdout prints (the periodic probe calls this many times).
    """

    if verbose:
        print('Start evaluation for model {}.'.format(counter))

    # device=None keeps the historical behaviour (temp TestWorker model copy);
    # the in-training per-regime reward probe passes the LIVE model's device so
    # test() never yanks the training model onto CPU (selfplay_on_gpu is False).
    if device is None:
        device = 'cuda' if (config.selfplay_on_gpu and torch.cuda.is_available()) else 'cpu'
    model.to(device)
    model.eval()

    # new games
    if envs is None:
        envs = [config.new_game(seed=i, save_video=save_video) for i in range(test_episodes)]
        create_temp_envs = True
    else:
        create_temp_envs = False

    with torch.no_grad():
        max_episode_steps = envs[0].get_max_episode_steps()
        # initializations. pin_g may be None (env samples its own regime), a
        # scalar (all envs -> that regime), or a per-env sequence (one regime
        # id per env -> a single mixed-regime batch search, which the per-regime
        # reward probe uses to keep the batch wide + GPU-efficient).
        if pin_g is None:
            init_obses = [env.reset() for env in envs]
        elif isinstance(pin_g, (list, tuple, np.ndarray)):
            init_obses = [env.reset(g=int(pin_g[i])) for i, env in enumerate(envs)]
        else:
            init_obses = [env.reset(g=int(pin_g)) for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(config=config, ray_store_obs=False) for _ in range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

        step = 0
        eps_steps_lst = np.zeros(test_episodes)
        eps_reward_lst = np.zeros(test_episodes)

        # subjective model: per-episode belief-GRU hiddens (pure inference —
        # evaluation never sees the oracle regime id)
        belief_hiddens = None
        if getattr(config, "subjective_model", False):
            belief_hiddens = torch.zeros(
                test_episodes, config.num_agents, model.belief_net.gru_hidden,
                device=device,
            )

        if config.case in ['smac', 'gfootball']:
            battle_won_lst = np.zeros(test_episodes)

        # loop
        while (not dones.all()) and step < max_episode_steps:

            stack_obs = [game_history.step_obs() for game_history in game_histories]
            stack_obs = prepare_observation_lst(stack_obs, config.image_based)
            if config.image_based:
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = torch.from_numpy(stack_obs).to(device).float()

            if belief_hiddens is not None:
                obs_flat = stack_obs.reshape(test_episodes, config.num_agents, -1)
                belief_hiddens, g_hat = model.belief_net.step(obs_flat, belief_hiddens)
                model.set_belief(g_hat)

            with autocast():
                network_output = model.initial_inference(stack_obs.float())
            legal_actions_lst = np.asarray([env.legal_actions() for env in envs])

            if config.use_mcts_test:
                search_results = SampledMCTS(config, np_random).batch_search(model, network_output, legal_actions_lst, device, False, 1.0)

                roots_sampled_visit_counts = search_results.sampled_visit_count
                roots_sampled_actions = search_results.sampled_actions
            else:
                # use network output directly as the evaluation policy, instead of MCTS search
                batch_policy_logits = network_output.policy_logits
                batch_policy_probs = np.exp(batch_policy_logits - np.max(batch_policy_logits, axis=-1, keepdims=True))
                batch_policy_probs *= legal_actions_lst
                batch_policy_probs = batch_policy_probs / np.sum(batch_policy_probs, axis=-1, keepdims=True)    # type: np.ndarray

            for i in range(test_episodes):
                if dones[i]:
                    continue

                # select the argmax, not sampling
                if config.use_mcts_test:
                    action_pos, _ = select_action(
                        roots_sampled_visit_counts[i],
                        temperature=1,
                        deterministic=True,
                        np_random=np_random
                    )
                    action = roots_sampled_actions[i][action_pos]
                else:
                    action = np.argmax(batch_policy_probs[i], axis=-1)

                next_obs, reward, done, info = envs[i].step(action)
                dones[i] = done
                eps_steps_lst[i] += 1
                eps_reward_lst[i] += float(np.sum(reward) if sum_agents
                                           else np.mean(reward))

                game_histories[i].store_transition(action, reward, next_obs)

                if config.case in ['smac', 'gfootball']:
                    battle_won_lst[i] = info['battle_won']

            step += 1

    if create_temp_envs:
        for env in envs:
            env.close()

    test_logs = {
        'test_counter': counter,
        'mean_score': eps_reward_lst.mean(),
        'std_score': eps_reward_lst.std(),
        'max_score': eps_reward_lst.max(),
        'min_score': eps_reward_lst.min(),
    }
    if sum_agents:
        # per-episode team returns for the per-regime reward probe. Gated on
        # sum_agents (only the probe sets it) because the default test_worker
        # path forwards test_logs to core/log.py::_log, which logs one SCALAR
        # per key — an array here crashes that loop. The probe reads this back
        # directly and never forwards test_logs to _log.
        test_logs['scores'] = eps_reward_lst.copy()
    if config.case in ['smac', 'gfootball']:
        test_logs['win_rate'] = np.mean(battle_won_lst)

    test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
               ''.format(test_logs['test_counter'], config.env_name, test_logs["mean_score"], test_logs["max_score"], test_logs["min_score"], test_logs["std_score"])
    if 'win_rate' in test_logs:
        test_msg += ' | WinRate: {:.2f}'.format(test_logs['win_rate'])
    if verbose:
        print(test_msg)

    return test_logs, step


class TestWorker(object):

    def __init__(self, config: BaseConfig):
        self.config = config
        self.eval_model = config.get_uniform_network()
        self.np_random, _ = seeding.np_random(config.seed * 3000)

        self.test_episodes = config.test_episodes
        self.eval_envs = [config.new_game(seed=i, save_video=False) for i in range(config.test_episodes)]

        self.device = 'cuda' if (config.selfplay_on_gpu and torch.cuda.is_available()) else 'cpu'
        self.eval_model.to(self.device)
        self.eval_model.eval()
        self.last_model_index = -1

    def update_model(self, model_index, weights):
        self.eval_model.set_weights(weights)
        self.last_model_index = model_index

    def run(self):
        """run evaluation test once
        """
        return test(self.config, self.eval_model, self.last_model_index, self.test_episodes, self.eval_envs, self.np_random)

    def close(self):
        for env in self.eval_envs:
            env.close()


@ray.remote
class RemoteTestWorker(TestWorker):

    def __init__(self, config: BaseConfig, shared_storage):
        super().__init__(config)
        self.shared_storage = shared_storage

    def run_loop(self):
        best_test_score = float('-inf')
        while True:
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(10)
                break
            if self.last_model_index // self.config.test_interval < trained_steps // self.config.test_interval:
                model_index, weights = ray.get(self.shared_storage.get_weights.remote())
                self.update_model(model_index, weights)

                test_log, eval_steps = self.run()

                self.shared_storage.add_test_logs.remote(test_log)

                if test_log['mean_score'] >= best_test_score:
                    best_test_score = test_log['mean_score']
                    torch.save(self.eval_model.state_dict(), self.config.model_path)

            time.sleep(30)

        self.close()
