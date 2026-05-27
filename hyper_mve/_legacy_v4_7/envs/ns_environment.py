"""
Non-Stationary MultiAgentEnv

Extends the base MultiAgentEnv to expose the current Rule through:
    - reset(options=None) returns (obs_n, rule)
      options={'rule': float} to force a specific rule for the episode
    - step() includes rule in info dict
    - get_rule() method for direct access
"""
from multiagent.environment import MultiAgentEnv


class NonStationaryMultiAgentEnv(MultiAgentEnv):
    """MultiAgentEnv wrapper that exposes the environment Rule."""

    def reset(self, options=None):
        """
        Reset environment.

        Args:
            options: Optional dict. If options={'rule': val}, force
                     that rule for this episode (instead of random sampling).

        Returns:
            (obs_n, rule)
        """
        # Set force_rule on the world BEFORE calling super().reset(),
        # so that Scenario.reset_world() can pick it up.
        if options is not None and 'rule' in options:
            self.world.force_rule = options['rule']
        else:
            self.world.force_rule = None

        obs_n = super().reset()
        return obs_n, self.get_rule()

    def step(self, action_n):
        """Step environment. Adds 'rule' to info dict."""
        obs_n, reward_n, done_n, info_n = super().step(action_n)
        info_n['rule'] = self.get_rule()
        return obs_n, reward_n, done_n, info_n

    def get_rule(self):
        """Return the current Rule value from the world."""
        return self.world.rule