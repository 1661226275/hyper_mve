import os
import cv2
import gymnasium as gym
import torch
import random
import shutil
import logging
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from scipy.stats import entropy


# global remote actor handles
remote_worker_handles = []


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def get_max_episode_steps(self):
        return self._max_episode_steps

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        img = self.max_frame
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


def make_atari(env_id, skip=4, max_episode_steps=None):
    """Make Atari games
    Parameters
    ----------
    env_id: str
        name of environment
    skip: int
        frame skip
    max_episode_steps: int
        max moves for an episode
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        print('Warning, path exists! Rewriting...')
        shutil.rmtree(exp_path)
        os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    # initialize the logger
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def select_action(visit_counts, temperature=1, deterministic=True, np_random: np.random.RandomState = None):
    """select action from the root visit counts.

    Parameters
    ----------
    visit_counts: list[int]
        visit counts
    temperature: float
        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    assert sum(visit_counts) > 0, "invalid input: num_simulation = 0!!!"
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]

    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]

    action_probs = np.array(action_probs)

    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        if np_random is not None:
            action_pos = np_random.choice(len(visit_counts), p=action_probs)
        else:
            action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy


def eps_greedy_action(greedy_action, legal_action_mask, eps):
    greedy_action = torch.from_numpy(greedy_action)
    legal_action_mask = torch.from_numpy(legal_action_mask)

    random_numbers = torch.rand_like(legal_action_mask[..., 0].float())
    pick_random = (random_numbers < eps).long()
    random_actions = torch.distributions.Categorical(legal_action_mask).sample()

    res = pick_random * random_actions + (1 - pick_random) * greedy_action

    return res.numpy()


def prepare_observation_lst(observation_lst, image_based):
    """Prepare the observations to satisfy the input fomat of torch
    [B, S, N, W, H, C] -> [B, N, S x C, W, H]
    batch, stack num, agents num, width, height, channel
    """
    if image_based:
        # B, S, W, H, C
        observation_lst = np.array(observation_lst, dtype=np.uint8)
    else:
        observation_lst = np.array(observation_lst, dtype=np.float32)
    observation_lst = np.moveaxis(observation_lst, [2, 1, -1], [1, 2, 3])
    # observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], shape[1], -1, shape[-2], shape[-1]))
    # observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def get_max_entropy(action_space_size):
    p = 1.0 / action_space_size
    ep = - action_space_size * p * np.log2(p)
    return ep


def arr_to_str(arr):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str


def str_to_arr(s, gray_scale=False):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function decodes the observation numpy arr from the jpeg strings
    Parameters
    ----------
    s: string
        the inputs
    gray_scale: bool
        True -> the inputs observation is gray not RGB.
    """
    nparr = np.frombuffer(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr


def concat_with_zero_padding(lst: List[np.ndarray], max_len: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """concat a list of numpy arrays with different length, padding to max_length and fill in zero.

    Parameters
    ----------
    lst : List[np.ndarray]
    max_len : int
        equals to max(len(lst[i])) if not set

    Returns
    -------
    (arr: np.ndarray, mask: np.ndarray)
    """
    len_lst = np.array([len(x) for x in lst])
    num_arr = len_lst.shape[0]
    max_len = len_lst.max() if max_len is None else max_len
    arr = np.zeros((num_arr, max_len, *lst[0].shape[1:]), dtype=lst[0].dtype)
    mask = np.arange(max_len) < len_lst.reshape(-1, 1)
    arr[mask] = np.concatenate(lst)
    return arr, mask


def profile(func):
    from line_profiler import LineProfiler

    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()

        return result
    return wrapper


class Timer:
    def __init__(self):
        self.start_times = defaultdict(float)
        self.durations = defaultdict(float)
        self.block_counts = defaultdict(int)

    def start(self, block_name='default'):
        self.start_times[block_name] = time.time()
        self.block_counts[block_name] += 1

    def stop(self, block_name='default'):
        self.durations[block_name] += time.time() - self.start_times[block_name]

    def avg(self, block_name='default'):
        return self.durations[block_name] / (self.block_counts[block_name] + 1e-8)

    def sum(self, block_name='default'):
        return self.durations[block_name]

    def reset(self):
        self.start_times.clear()
        self.durations.clear()
        self.block_counts.clear()


def policy_target_informative(adv_norm, masks, visit_policies,
                              min_qstd=0.0, min_children=2):
    """Which transitions carry a policy target worth learning from? -> (rows,)

    Stage C of the prior-collapse fix — the modern analogue of the retired
    architecture's ``planner_on`` mask, which excluded self-distilled targets
    from the policy CE ("the zero-information fixed point").

    Under ``--policy_target_type q_softmax`` the target is
    ``softmax(adv/temp)`` over the root's children. When those advantages are
    flat, that softmax is near-UNIFORM — and a uniform target is not a weak
    gradient, it is a gradient actively pushing the policy toward uniform. The
    retired planner guarded the same case with ``mve_qstd_floor`` (uniform
    fallback when candidate returns are noise-level flat).

    Measured live: in relation/g1 (``mutual_comp``, zero-sum) the advantage
    spread is ~0.29 of the batch-pooled spread on the main method, so ~20% of
    training transitions would contribute near-uniform targets.

    SCOPE NOTE (2026-08-09): that measurement was taken on the v5 ``g2`` family,
    and ``mutual_comp`` is not in ``g2cm`` — the zero-sum regime that motivated
    this threshold no longer exists there. The guard is left as-is rather than
    retuned on a guess: it is scale-free by construction, so it should degrade
    gracefully, but the "~20% of transitions" figure does not carry over and
    the threshold is unvalidated on ``g2cm`` until re-measured.

    ``adv_norm`` is ``(rows, C, N)`` ALREADY divided by the batch-pooled
    per-agent ``adv_std``. That is what makes ``min_qstd`` scale-free: a value
    of 0.4 means "this root's advantage spread is under 40% of the batch-typical
    spread", which transfers across architectures and across training as the
    value head sharpens. An ABSOLUTE threshold would not: the Bayes-averaged
    head runs 2.6x wider than the single head, so one fixed number masks g1
    alone on one arm and every regime on the other.

    ``min_qstd <= 0`` disables the guard, returning all-ones — bit-exact with
    not applying it at all.
    """
    import warnings

    import numpy as np

    rows = adv_norm.shape[0]
    if min_qstd <= 0:
        return np.ones(rows, dtype=np.float32)

    m = np.asarray(masks, dtype=bool)                       # (rows, C)
    # a root with too few live children cannot express a preference at all
    n_children = m.sum(axis=1)                              # (rows,)
    # ...and children that were never visited carry no evaluation
    n_visited = ((np.asarray(visit_policies) > 0) & m).sum(axis=1)

    a = np.array(adv_norm, dtype=np.float64)
    a[~m] = np.nan
    # A fully-masked row is an all-NaN slice: nanstd warns and returns NaN,
    # which nan_to_num maps to 0 (and n_children == 0 rejects it anyway).
    # Expected and handled, so don't let it spam the training log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # spread over the sampled children, then the agent-mean; a target is
        # uninformative only when it is flat for EVERY agent, so the mean (not
        # the min) is the right reduction.
        row_std = np.nanstd(a, axis=1).mean(axis=-1)        # (rows,)
    row_std = np.nan_to_num(row_std, nan=0.0)

    ok = (n_children >= min_children) & (n_visited >= 2) & (row_std >= min_qstd)
    return ok.astype(np.float32)
