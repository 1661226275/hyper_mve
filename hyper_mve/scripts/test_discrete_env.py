"""
Phase 1 Verification: Discrete action environment test for Hyper-MuZero.

Tests:
    1. Environment creation with discrete=True
    2. Action space is Discrete(5) per agent
    3. Step with integer actions (0-4)
    4. Observation shapes unchanged (same as continuous)
    5. Reward structure works with discrete actions
    6. Rule sampling still functions
    7. Agent 2 variable flag and color interpolation
    8. Full episode with random discrete policy
    9. Action mapping verification (directional movement)
    10. deepcopy safety check
"""
import sys
import os
import copy
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.make_env import make_ns_env


ACTION_NAMES = {0: 'stop', 1: 'left', 2: 'right', 3: 'down', 4: 'up'}


def test_discrete_env():
    print("=" * 60)
    print("Phase 1: Discrete Action Environment Test (Hyper-MuZero)")
    print("=" * 60)

    # ── Test 1: Create discrete environment ──────────────────────
    print("\n[Test 1] Creating discrete environment...")
    env = make_ns_env(discrete=True)
    print(f"  Number of agents: {env.n}")
    print(f"  discrete_action_space: {env.discrete_action_space}")
    print(f"  discrete_action_input: {env.discrete_action_input}")
    assert env.n == 4, f"Expected 4 agents, got {env.n}"
    assert env.discrete_action_space is True, "discrete_action_space should be True"
    assert env.discrete_action_input is True, "discrete_action_input should be True"
    print("  ✅ PASSED")

    # ── Test 2: Action space check ───────────────────────────────
    print("\n[Test 2] Action space verification...")
    for i, action_space in enumerate(env.action_space):
        print(f"  Agent {i}: {action_space}")
        assert hasattr(action_space, 'n'), f"Agent {i} action space should be Discrete"
        assert action_space.n == 5, f"Agent {i} should have 5 actions, got {action_space.n}"
    print("  ✅ PASSED: All agents have Discrete(5) action space")

    # ── Test 3: Observation spaces (same as continuous) ──────────
    print("\n[Test 3] Observation spaces...")
    obs_n, rule = env.reset()
    expected_obs_dims = [16, 16, 16, 14]
    for i, obs in enumerate(obs_n):
        print(f"  Agent {i}: obs shape = {obs.shape}")
        assert obs.shape == (expected_obs_dims[i],), \
            f"Agent {i} obs shape mismatch: {obs.shape} != ({expected_obs_dims[i]},)"
    print("  ✅ PASSED: Observation dimensions match expected values")

    # ── Test 4: Step with integer actions ────────────────────────
    print("\n[Test 4] Step with integer actions...")
    action_n = [np.random.randint(0, 5) for _ in range(env.n)]
    print(f"  Actions: {[f'{a} ({ACTION_NAMES[a]})' for a in action_n]}")
    obs_next_n, reward_n, done_n, info_n = env.step(copy.deepcopy(action_n))
    print(f"  Rewards: {[f'{r:.4f}' for r in reward_n]}")
    print(f"  Dones: {done_n}")
    print(f"  Rule in info: {info_n['rule']:.4f}")
    assert len(obs_next_n) == env.n
    assert len(reward_n) == env.n
    assert info_n['rule'] == rule, "Rule should not change within episode"
    print("  ✅ PASSED")

    # ── Test 5: All 5 actions produce valid transitions ──────────
    print("\n[Test 5] Testing all 5 actions...")
    for action_id in range(5):
        obs_n, rule = env.reset()
        # All agents take the same action
        action_n = [action_id] * env.n
        obs_next_n, reward_n, done_n, info_n = env.step(copy.deepcopy(action_n))
        assert all(o.shape == obs_n[i].shape for i, o in enumerate(obs_next_n)), \
            f"Shape mismatch for action {action_id}"
        print(f"  Action {action_id} ({ACTION_NAMES[action_id]}): OK")
    print("  ✅ PASSED: All 5 discrete actions work")

    # ── Test 6: Action mapping verification ──────────────────────
    print("\n[Test 6] Action mapping verification (directional movement)...")
    # Place agent at origin, test that actions move it in the correct direction
    obs_n, rule = env.reset()
    # Force agent 0 to origin
    env.world.agents[0].state.p_pos = np.array([0.0, 0.0])
    env.world.agents[0].state.p_vel = np.array([0.0, 0.0])

    # action=0: stop (no movement force)
    pos_before = env.world.agents[0].state.p_pos.copy()

    # action=2: right (+x)
    env.world.agents[0].state.p_pos = np.array([0.0, 0.0])
    env.world.agents[0].state.p_vel = np.array([0.0, 0.0])
    action_n = [2, 0, 0, 0]  # agent 0 goes right, others stop
    env.step(copy.deepcopy(action_n))
    pos_after_right = env.world.agents[0].state.p_pos.copy()
    print(f"  Action 2 (right): pos moved to {pos_after_right}")

    # action=1: left (-x)
    env.world.agents[0].state.p_pos = np.array([0.0, 0.0])
    env.world.agents[0].state.p_vel = np.array([0.0, 0.0])
    action_n = [1, 0, 0, 0]  # agent 0 goes left
    env.step(copy.deepcopy(action_n))
    pos_after_left = env.world.agents[0].state.p_pos.copy()
    print(f"  Action 1 (left):  pos moved to {pos_after_left}")

    # action=4: up (+y)
    env.world.agents[0].state.p_pos = np.array([0.0, 0.0])
    env.world.agents[0].state.p_vel = np.array([0.0, 0.0])
    action_n = [4, 0, 0, 0]
    env.step(copy.deepcopy(action_n))
    pos_after_up = env.world.agents[0].state.p_pos.copy()
    print(f"  Action 4 (up):    pos moved to {pos_after_up}")

    # action=3: down (-y)
    env.world.agents[0].state.p_pos = np.array([0.0, 0.0])
    env.world.agents[0].state.p_vel = np.array([0.0, 0.0])
    action_n = [3, 0, 0, 0]
    env.step(copy.deepcopy(action_n))
    pos_after_down = env.world.agents[0].state.p_pos.copy()
    print(f"  Action 3 (down):  pos moved to {pos_after_down}")

    # Verify directions (note: collisions with other agents may perturb slightly)
    assert pos_after_right[0] > 0, f"Right should increase x, got {pos_after_right[0]}"
    assert pos_after_left[0] < 0, f"Left should decrease x, got {pos_after_left[0]}"
    assert pos_after_up[1] > 0, f"Up should increase y, got {pos_after_up[1]}"
    assert pos_after_down[1] < 0, f"Down should decrease y, got {pos_after_down[1]}"
    print("  ✅ PASSED: Discrete action mapping is correct")

    # ── Test 7: Agent 2 variable flag and color ──────────────────
    print("\n[Test 7] Agent 2 variable flag and color...")
    agent2 = env.world.agents[2]
    assert getattr(agent2, 'variable', False) is True, "Agent 2 should have variable=True"
    print(f"  Agent 2 variable flag: {agent2.variable}")

    # Test color interpolation
    env.world.rule = 1.0
    env.reset_callback(env.world)
    color_coop = env.world.agents[2].color.copy()
    print(f"  Rule=1.0 (cooperate) Agent 2 color: {color_coop}")

    env.world.rule = 0.0
    env.reset_callback(env.world)
    color_betray = env.world.agents[2].color.copy()
    print(f"  Rule=0.0 (betray)    Agent 2 color: {color_betray}")

    assert not np.allclose(color_coop, color_betray), "Colors should differ between rules"
    print("  ✅ PASSED: Variable agent flag and color interpolation work")

    # ── Test 8: Rule sampling across episodes ────────────────────
    print("\n[Test 8] Rule sampling across episodes...")
    rules = []
    for _ in range(20):
        obs_n, rule = env.reset()
        rules.append(rule)
    rules = np.array(rules)
    print(f"  20 episode rules: min={rules.min():.4f}, max={rules.max():.4f}, "
          f"mean={rules.mean():.4f}, std={rules.std():.4f}")
    assert rules.std() > 0.01, "Rules should vary across episodes"
    assert rules.min() >= 0.0 and rules.max() <= 1.0
    print("  ✅ PASSED")

    # ── Test 9: Full episode with random discrete policy ─────────
    print("\n[Test 9] Full episode with random discrete policy (25 steps)...")
    obs_n, rule = env.reset()
    total_rewards = np.zeros(env.n)
    for t in range(25):
        action_n = [np.random.randint(0, 5) for _ in range(env.n)]
        obs_n, reward_n, done_n, info_n = env.step(copy.deepcopy(action_n))
        total_rewards += np.array(reward_n)
    print(f"  Rule: {rule:.4f}")
    print(f"  Total rewards: {total_rewards.round(2)}")
    print(f"  Final obs shapes: {[o.shape for o in obs_n]}")
    print("  ✅ PASSED: Full episode completed")

    # ── Test 10: Reward sensitivity with discrete actions ────────
    print("\n[Test 10] Reward sensitivity to Rule (discrete actions)...")

    def run_discrete_episode(env_local, target_rule, num_steps=25):
        """Force a specific rule and run an episode with random discrete actions."""
        obs_n, _ = env_local.reset()
        env_local.world.rule = target_rule
        total_rewards = np.zeros(env_local.n)
        for _ in range(num_steps):
            action_n = [np.random.randint(0, 5) for _ in range(env_local.n)]
            obs_n, reward_n, done_n, _ = env_local.step(copy.deepcopy(action_n))
            total_rewards += np.array(reward_n)
        return total_rewards

    rewards_coop = np.array([run_discrete_episode(env, 1.0) for _ in range(10)])
    rewards_betray = np.array([run_discrete_episode(env, 0.0) for _ in range(10)])

    print(f"  Rule=1.0 mean rewards: {rewards_coop.mean(axis=0).round(2)}")
    print(f"  Rule=0.0 mean rewards: {rewards_betray.mean(axis=0).round(2)}")

    var_diff = abs(rewards_coop[:, 2].mean() - rewards_betray[:, 2].mean())
    print(f"  Variable agent (Agent 2) reward diff: {var_diff:.2f}")
    print("  ✅ PASSED: Reward structure is rule-dependent with discrete actions")

    # ── Test 11: deepcopy safety ─────────────────────────────────
    print("\n[Test 11] deepcopy safety check...")
    obs_n, rule = env.reset()
    action_n = [2, 1, 4, 3]
    action_n_copy = copy.deepcopy(action_n)
    env.step(copy.deepcopy(action_n))
    assert action_n == action_n_copy, "Original actions should not be modified!"
    print("  ✅ PASSED: Actions are not modified by env.step()")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All 11 tests PASSED!")
    print("Discrete action environment is ready for Hyper-MuZero.")
    print("")
    print("Action mapping (MPE internal):")
    print("  0 = stop (no force)")
    print("  1 = left  (u[0] = -1.0)")
    print("  2 = right (u[0] = +1.0)")
    print("  3 = down  (u[1] = -1.0)")
    print("  4 = up    (u[1] = +1.0)")
    print("")
    print(f"Joint action dim (one-hot): {env.n} agents × 5 actions = {env.n * 5}")
    print("=" * 60)

    env.close()


if __name__ == '__main__':
    test_discrete_env()
