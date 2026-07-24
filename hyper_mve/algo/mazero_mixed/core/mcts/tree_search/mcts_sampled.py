from typing import List, NamedTuple, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast as autocast

from core.model import BaseNet, NetworkOutput
from core.config import BaseConfig
from core.mcts.ctree.ctree_sampled import cytree


class SearchOutput(NamedTuple):
    value: np.ndarray                           # search root value (agent-mean): (batch_size, )
    marginal_visit_count: np.ndarray
    marginal_priors: np.ndarray
    sampled_actions: List[np.ndarray]           # sampled joint actions:    [(K1, num_agents), (K2, num_agents), ...]
    sampled_visit_count: List[np.ndarray]       # sampled visit count:      [(K1,), (K2,), ...]
    sampled_pred_probs: List[np.ndarray]
    sampled_beta: List[np.ndarray]
    sampled_beta_hat: List[np.ndarray]
    sampled_priors: List[np.ndarray]
    sampled_imp_ratio: List[np.ndarray]
    sampled_pred_values: List[np.ndarray]
    sampled_mcts_values: List[np.ndarray]
    sampled_rewards: List[np.ndarray]
    sampled_qvalues: List[np.ndarray]
    value_vec: np.ndarray = None                # per-agent root value: (batch_size, num_agents)
    sampled_qvalues_vec: List[np.ndarray] = None  # per-agent: [(K1, num_agents), ...]


class SampledMCTS(object):
    def __init__(self, config: BaseConfig, np_random: np.random.RandomState = None):
        self.config = config
        self.np_random = np.random if np_random is None else np_random

    def batch_search(
        self,
        model: BaseNet,
        network_output: NetworkOutput,
        legal_actions_lst: np.ndarray = None,
        device: torch.device = None,
        add_noise: bool = False,
        sampled_tau: float = 1.0,
        sampled_actions_res: Tuple[np.ndarray, np.ndarray] = None,
    ) -> SearchOutput:
        """Create a batch of root nodes from network_output and do MCTS in parallel.
        """
        pb_c_base, pb_c_init, discount, rho, lam = self.config.pb_c_base, self.config.pb_c_init, self.config.discount, self.config.mcts_rho, self.config.mcts_lambda
        noise_alpha, noise_epsilon = self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
        num_agents, action_space_size, sampled_times = self.config.num_agents, self.config.action_space_size, self.config.sampled_action_times
        batch_size = network_output.hidden_state.shape[0]

        batch_hidden_states = network_output.hidden_state           # (batch_size, state_hidden_size)
        batch_rewards = network_output.reward                       # (batch_size, 1)
        batch_values = network_output.value                         # (batch_size, 1)
        batch_policy_logits = network_output.policy_logits          # (batch_size, num_agents, action_space_size)
        # value may be a team scalar (B, 1) or per-agent (B, N) / (B, N, 1) / (B, N, S)-collapsed
        assert batch_values.reshape(batch_size, -1).shape[1] in (1, num_agents) and batch_policy_logits.shape == (batch_size, num_agents, action_space_size)

        def widen(x):
            # broadcast a team scalar per root to a per-agent vector; pass
            # per-agent vectors through unchanged. -> (batch_size, num_agents)
            x = x.reshape(batch_size, -1)
            if x.shape[1] == 1:
                x = np.repeat(x, num_agents, axis=1)
            assert x.shape == (batch_size, num_agents)
            return np.ascontiguousarray(x, dtype=np.float32)

        batch_policy_probs = np.exp(batch_policy_logits - np.max(batch_policy_logits, axis=-1, keepdims=True))
        batch_policy_probs = batch_policy_probs / np.sum(batch_policy_probs, axis=-1, keepdims=True)

        # exploration noise
        noises = self.np_random.dirichlet([noise_alpha] * action_space_size, batch_size * num_agents).astype(np.float32).reshape(batch_size, num_agents, action_space_size)
        if not add_noise:
            noise_epsilon = 0.0
        
        # check if need legal action mask
        if legal_actions_lst is not None:
            batch_policy_probs *= legal_actions_lst
            # avoid zero denominator when normalized
            batch_policy_probs += legal_actions_lst * 1e-4
            assert ~(np.sum(batch_policy_probs, axis=-1) == 0).sum()
            batch_policy_probs = batch_policy_probs / np.sum(batch_policy_probs, axis=-1, keepdims=True)

            noises *= legal_actions_lst
            noises += legal_actions_lst * 1e-4
            noises = noises / np.sum(noises, axis=-1, keepdims=True)

        # the data storage of hidden states: storing the states of all the tree nodes
        hidden_states_pool = [batch_hidden_states]         # type: List[torch.Tensor]
        # initialize MCTS tree (select_mode: 0 = joint team-UCB, 1 = decoupled per-agent UCB)
        select_mode = 1 if getattr(self.config, "decoupled_selection", False) else 0
        root_cover_mode = int(getattr(self.config, "root_cover_mode", 0))
        # Leaf expansion keeps its own (smaller) sample budget. sampled_times
        # doubles as the replay-buffer width, which the root cover forces up to
        # 1 + N*A; feeding that same number to every leaf would multiply
        # per-node selection work and node-pool usage for no benefit -- and the
        # cost would appear exactly as the policy de-collapses and leaf draws
        # stop deduping. 0 means "same as sampled_times" (upstream behaviour).
        leaf_sampled_times = int(getattr(self.config, "leaf_sampled_times", 0)) or sampled_times
        trees = cytree.Tree_batch(batch_size, num_agents, action_space_size, sampled_times, self.config.num_simulations, self.config.tree_value_stat_delta_lb, self.np_random.choice(256), rho, lam, select_mode, root_cover_mode)

        # (a) prepare root node with re-sampled actions
        if sampled_actions_res is None:
            batch_beta = batch_policy_probs * (1 - noise_epsilon) + noises * noise_epsilon
            batch_beta = batch_beta ** (1 / sampled_tau)
            # mask policy_probs with legal_action_lst if given
            if legal_actions_lst is not None:
                batch_beta *= legal_actions_lst
                # avoid zero denominator when normalized
                assert ~(np.sum(batch_beta, axis=-1) == 0).sum()
            batch_beta = batch_beta / np.sum(batch_beta, axis=-1, keepdims=True)

            batch_rewards = widen(batch_rewards)
            batch_values = widen(batch_values)
            batch_policy_probs = batch_policy_probs.astype(np.float32)
            batch_beta = batch_beta.astype(np.float32)
            trees.prepare(batch_rewards, batch_values, batch_policy_probs, batch_beta, sampled_times, noise_epsilon, noises)
        # (b) prepare root node with fixed actions set
        else:
            raise NotImplementedError

        with torch.no_grad():
            model.eval()    # ensure model.output datatype: np.ndarray

            for index_simulation in range(self.config.num_simulations):
                batch_hidden_states = []

                # traverse to select actions for each root
                hidden_state_index_x_lst, hidden_state_index_y_lst, batch_actions = \
                    trees.batch_selection(pb_c_base, pb_c_init, discount)

                # obtain the states for leaf nodes at the end of current simulation
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    batch_hidden_states.append(hidden_states_pool[ix][iy])      # append shape: hidden_state_shape

                # convert search results into torch tensor
                batch_hidden_states = torch.vstack(batch_hidden_states)
                batch_actions = torch.from_numpy(batch_actions).to(device)

                # evaluation for leaf nodes
                with autocast():
                    network_output = model.recurrent_inference(batch_hidden_states, batch_actions)

                batch_hidden_states = network_output.hidden_state       # (batch_size, state_hidden_size)
                batch_rewards = network_output.reward                   # (batch_size, 1)
                batch_values = network_output.value                     # (batch_size, 1)
                batch_policy_logits = network_output.policy_logits      # (batch_size, num_agents, action_space_size)

                batch_policy_probs = np.exp(batch_policy_logits - np.max(batch_policy_logits, axis=-1, keepdims=True))
                batch_policy_probs = batch_policy_probs / np.sum(batch_policy_probs, axis=-1, keepdims=True)    # type: np.ndarray
                batch_beta = batch_policy_probs ** (1 / sampled_tau)
                batch_beta = batch_beta / np.sum(batch_beta, axis=-1, keepdims=True)

                # save leaf nodes' states into states pool
                hidden_states_pool.append(batch_hidden_states)

                # expand the leaf nodes and backup along the search path to update the attributes
                batch_rewards = widen(batch_rewards)
                batch_values = widen(batch_values)
                batch_policy_probs = batch_policy_probs.astype(np.float32)
                batch_beta = batch_beta.astype(np.float32)
                trees.batch_expansion_and_backup(index_simulation + 1, discount, leaf_sampled_times,
                                                 batch_rewards, batch_values, batch_policy_probs, batch_beta)

        # get target value/policy from MCTS results
        roots_values = trees.get_roots_values()                             # (batch_size, )

        roots_marginal_visit_count = trees.get_roots_marginal_visit_count()  # (batch_size, num_agents, action_space_size)
        roots_marginal_priors = trees.get_roots_marginal_priors()

        roots_sampled_actions = trees.get_roots_sampled_actions()           # list of batch_size, each element has shape (K1, num_agents), (K2, num_agents), ...
        roots_sampled_visit_count = trees.get_roots_sampled_visit_count()   # list of batch_size, each element has shape (K1,), (K2,), ... K <= sampled_times
        roots_sampled_pred_probs = trees.get_roots_sampled_pred_probs()
        roots_sampled_beta = trees.get_roots_sampled_beta()
        roots_sampled_beta_hat = trees.get_roots_sampled_beta_hat()
        roots_sampled_priors = trees.get_roots_sampled_priors()
        roots_sampled_imp_ratio = trees.get_roots_sampled_imp_ratio()
        roots_sampled_pred_values = trees.get_roots_sampled_pred_values()
        roots_sampled_mcts_values = trees.get_roots_sampled_mcts_values()
        roots_sampled_rewards = trees.get_roots_sampled_rewards()
        roots_sampled_qvalues = trees.get_roots_sampled_qvalues(discount)
        roots_values_vec = trees.get_roots_values_vec()                      # (batch_size, num_agents)
        roots_sampled_qvalues_vec = trees.get_roots_sampled_qvalues_vec(discount)

        # assert np.all(np.sum(roots_marginal_visit_count, axis=-1) == self.config.num_simulations)
        # assert np.all(np.sum(roots_sampled_visit_count, axis=-1) == self.config.num_simulations)
        # assert np.all(roots_sampled_rewards[0] + discount * roots_sampled_mcts_values[0] == roots_sampled_qvalues[0])

        return SearchOutput(roots_values, roots_marginal_visit_count, roots_marginal_priors, roots_sampled_actions, roots_sampled_visit_count,
                            roots_sampled_pred_probs, roots_sampled_beta, roots_sampled_beta_hat, roots_sampled_priors,
                            roots_sampled_imp_ratio,
                            roots_sampled_pred_values, roots_sampled_mcts_values, roots_sampled_rewards, roots_sampled_qvalues,
                            roots_values_vec, roots_sampled_qvalues_vec)
