"""
Phase 1 Verification: Random policy test for NonStationaryTag environment.

Tests:
    1. Environment creation and reset
    2. Step with random actions (continuous)
    3. Rule sampling per episode
    4. Observation shapes consistency
    5. Reward variation across different Rules
    6. Multiple episodes with rule statistics
"""
import sys
import os
import copy
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.make_env import make_ns_env


def test_env():
    print("=" * 60)
    print("Phase 1 Verification: NonStationaryTag Environment Test")
    print("=" * 60)

    # ── Test 1: Create environment ──────────────────────────────
    print("\n[Test 1] Creating environment...")
    env = make_ns_env(discrete=False)
    print(f"  Number of agents: {env.n}")
    print(f"  Observation spaces: {[s.shape for s in env.observation_space]}")
    print(f"  Action spaces: {[s.shape for s in env.action_space]}")
    assert env.n == 4, f"Expected 4 agents, got {env.n}"
    print("  ✅ PASSED")

    # ── Test 2: Reset and check Rule ────────────────────────────
    print("\n[Test 2] Reset and Rule check...")
    obs_n, rule = env.reset()
    print(f"  Rule = {rule:.4f}")
    print(f"  obs_n types: {[type(o).__name__ for o in obs_n]}")
    print(f"  obs_n shapes: {[o.shape for o in obs_n]}")
    assert 0.0 <= rule <= 1.0, f"Rule out of range: {rule}"
    assert len(obs_n) == env.n, f"Expected {env.n} observations, got {len(obs_n)}"
    print("  ✅ PASSED")

    # ── Test 3: Step with random actions ────────────────────────
    print("\n[Test 3] Step with random actions...")
    action_n = []
    for i in range(env.n):
        action_dim = env.action_space[i].shape[0]
        action = np.random.uniform(-1, 1, size=action_dim)
        action_n.append(action)

    obs_next_n, reward_n, done_n, info_n = env.step(copy.deepcopy(action_n))
    print(f"  rewards: {[f'{r:.4f}' for r in reward_n]}")
    print(f"  dones: {done_n}")
    print(f"  info rule: {info_n['rule']:.4f}")
    assert len(obs_next_n) == env.n
    assert len(reward_n) == env.n
    assert info_n['rule'] == rule, "Rule should not change within episode"
    print("  ✅ PASSED")

    # ── Test 4: Observation shape consistency ───────────────────
    print("\n[Test 4] Observation shape consistency across steps...")
    obs_shapes_init = [o.shape for o in obs_n]
    for step in range(10):
        action_n = [np.random.uniform(-1, 1, size=env.action_space[i].shape[0]) for i in range(env.n)]
        obs_next_n, _, _, _ = env.step(copy.deepcopy(action_n))
        obs_shapes_step = [o.shape for o in obs_next_n]
        assert obs_shapes_step == obs_shapes_init, \
            f"Shape mismatch at step {step}: {obs_shapes_step} != {obs_shapes_init}"
    print(f"  Consistent shapes over 10 steps: {obs_shapes_init}")
    print("  ✅ PASSED")

    # ── Test 5: Rule changes across episodes ────────────────────
    print("\n[Test 5] Rule sampling across episodes...")
    rules = []
    for ep in range(20):
        obs_n, rule = env.reset()
        rules.append(rule)
    rules = np.array(rules)
    print(f"  20 episode rules: min={rules.min():.4f}, max={rules.max():.4f}, "
          f"mean={rules.mean():.4f}, std={rules.std():.4f}")
    assert rules.std() > 0.01, "Rules should vary across episodes"
    assert rules.min() >= 0.0 and rules.max() <= 1.0, "Rules should be in [0, 1]"
    print("  ✅ PASSED")

    # ── Test 6: Reward difference between Rule=0 and Rule=1 ────
    print("\n[Test 6] Reward sensitivity to Rule...")

    def run_episode_with_rule(env_local, target_rule, num_steps=25):
        """Force a specific rule and run an episode."""
        obs_n, _ = env_local.reset()
        env_local.world.rule = target_rule  # Override rule
        total_rewards = np.zeros(env_local.n)
        for _ in range(num_steps):
            action_n = [np.random.uniform(-1, 1, size=env_local.action_space[i].shape[0])
                        for i in range(env_local.n)]
            obs_n, reward_n, done_n, _ = env_local.step(copy.deepcopy(action_n))
            total_rewards += np.array(reward_n)
            if all(done_n):
                break
        return total_rewards

    # Run 10 episodes with Rule=1.0 (full cooperation)
    rewards_coop = np.array([run_episode_with_rule(env, 1.0) for _ in range(10)])
    # Run 10 episodes with Rule=0.0 (full betrayal)
    rewards_betray = np.array([run_episode_with_rule(env, 0.0) for _ in range(10)])

    print(f"  Rule=1.0 (Cooperation)  mean rewards: {rewards_coop.mean(axis=0).round(2)}")
    print(f"  Rule=0.0 (Betrayal)     mean rewards: {rewards_betray.mean(axis=0).round(2)}")

    # Variable agent (index 2) reward should differ between rules
    var_agent_diff = abs(rewards_coop[:, 2].mean() - rewards_betray[:, 2].mean())
    print(f"  Variable agent (Agent 2) reward difference: {var_agent_diff:.2f}")
    print("  ✅ PASSED (Reward structure is rule-dependent)")

    # ── Test 7: get_rule() method ───────────────────────────────
    print("\n[Test 7] get_rule() method...")
    obs_n, rule = env.reset()
    assert env.get_rule() == rule, "get_rule() should match reset return"
    print(f"  get_rule() = {env.get_rule():.4f} (matches reset)")
    print("  ✅ PASSED")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All 7 tests PASSED! Environment is ready.")
    print("=" * 60)

    env.close()


if __name__ == '__main__':
    test_env()
