"""
Environment factory for Hyper-MuZero project.
Creates NonStationaryMultiAgentEnv with the non_stationary_tag scenario.

Default: discrete=True (5 actions per agent: stop/left/right/down/up)
"""
from envs.non_stationary_tag import Scenario
from envs.ns_environment import NonStationaryMultiAgentEnv


def make_ns_env(discrete=True, benchmark=False):
    """
    Create a Non-Stationary Tag environment.

    Args:
        discrete: If True, use discrete action space (5 actions).
                  Default True for Hyper-MuZero.
        benchmark: If True, include benchmark data callback.

    Returns:
        NonStationaryMultiAgentEnv instance.
    """
    scenario = Scenario()
    world = scenario.make_world()

    if benchmark:
        env = NonStationaryMultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            info_callback=scenario.benchmark_data,
            discrete=discrete,
        )
    else:
        env = NonStationaryMultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            discrete=discrete,
        )
    return env