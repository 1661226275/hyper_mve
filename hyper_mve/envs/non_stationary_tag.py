"""
Non-Stationary Tag Scenario — Faction Re-alignment (v3, Orthogonalized Rewards)

Based on simple_tag from MPE. Rule changes the global faction alignment,
affecting ALL agents' reward functions (not just Agent 2).

v3 Changes (Reward Orthogonalization):
    - Agent 2 guard mode: replaced -0.1*dist_to_prey with Blocking Point
      shielding reward (dense geometric guidance toward interception position)
    - Agent 2 guard mode: removed collision +1 with hunters (sparse + sign conflict)
    - Prey hunt mode: removed +0.1*dist_to_agent2 (eliminates sign cancellation
      with guard proximity at intermediate rule values)
    - Prey guard mode: removed +1.0 touching bonus (eliminates conflict with
      hunt collision -10)
    - All reward rows now have either same-sign or one-side-zero across
      r_hunt/r_guard — no signal cancellation at any rule value.

Agents:
    Agent 0, 1  - Fixed Hunters (always chase prey), color: red
    Agent 2     - Variable role (Hunter/Bodyguard depends on Rule), color: yellow
    Agent 3     - Prey (escape target), color: green

Rule mechanism (Faction Re-alignment):
    Rule ∈ [0, 1], sampled per episode, held constant within episode.

    Rule = 1 (Full Hunt):  {0,1,2} hunt {3}
    Rule = 0 (Bodyguard):  {0,1} hunt {3}, {2} shields {3}
    Intermediate Rule: Linear interpolation with orthogonal reward components.

Blocking Point mechanism (Guard mode, Agent 2):
    For each fixed hunter h_i, define blocking point:
        b_i = h_i.pos + alpha * (prey.pos - h_i.pos)   (alpha=0.6)
    Shield score = R_max * max(0, 1 - dist(agent2, b_i) / d_max)
    Multi-hunter aggregation: max over all hunters.
"""
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        world.dim_c = 2

        num_fixed_adversaries = 2
        num_variable_agents = 1
        num_good_agents = 1
        num_agents = num_fixed_adversaries + num_variable_agents + num_good_agents
        num_landmarks = 2

        # Add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

            if i < num_fixed_adversaries:
                # Fixed adversaries (Agent 0, 1)
                agent.adversary = True
                agent.variable = False
                agent.size = 0.075
                agent.accel = 3.0
                agent.max_speed = 1.0
            elif i < num_fixed_adversaries + num_variable_agents:
                # Variable role agent (Agent 2)
                agent.adversary = True  # Structurally an adversary
                agent.variable = True
                agent.size = 0.075
                agent.accel = 3.0
                agent.max_speed = 1.0
            else:
                # Prey (Agent 3)
                agent.adversary = False
                agent.variable = False
                agent.size = 0.05
                agent.accel = 4.0
                agent.max_speed = 1.3

        # Add landmarks (obstacles)
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        # Initialize rule (will be re-sampled in reset_world)
        world.rule = 0.5

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Use force_rule if set (from env.reset(options={'rule': val})),
        # otherwise sample randomly
        if hasattr(world, 'force_rule') and world.force_rule is not None:
            world.rule = world.force_rule
            world.force_rule = None  # consume once, reset to random next time
        else:
            world.rule = np.random.uniform(0.0, 1.0)

        # Set agent colors based on role
        for i, agent in enumerate(world.agents):
            if not agent.adversary:
                # Prey: green
                agent.color = np.array([0.35, 0.85, 0.35])
            elif getattr(agent, 'variable', False):
                # Variable agent: interpolate between blue (betray) and red (cooperate)
                r = world.rule
                agent.color = np.array([0.85 * r + 0.35 * (1 - r),
                                        0.85 * (1 - r) + 0.35 * r,
                                        0.35])
            else:
                # Fixed adversaries: red
                agent.color = np.array([0.85, 0.35, 0.35])

        # Landmark colors
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        # Random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    # ── Helper methods ──────────────────────────────────────────────────

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def good_agents(self, world):
        """Return prey agents."""
        return [a for a in world.agents if not a.adversary]

    def fixed_adversaries(self, world):
        """Return fixed adversary agents (not variable)."""
        return [a for a in world.agents if a.adversary and not getattr(a, 'variable', False)]

    def variable_agents(self, world):
        """Return variable role agents."""
        return [a for a in world.agents if getattr(a, 'variable', False)]

    def all_adversaries(self, world):
        """Return all adversary agents (fixed + variable)."""
        return [a for a in world.agents if a.adversary]

    # ── Helper: distance ────────────────────────────────────────────────

    def _dist(self, a1, a2):
        """Euclidean distance between two agents."""
        return np.sqrt(np.sum(np.square(a1.state.p_pos - a2.state.p_pos)))

    def _bound_penalty(self, agent, world):
        """Boundary penalty — same for all agents."""
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        penalty = 0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            penalty += bound(x)
        return penalty

    # ── Blocking Point shielding reward (v3) ────────────────────────────

    # Shielding reward parameters
    SHIELD_ALPHA = 0.6    # blocking point at 60% from hunter toward prey
    SHIELD_R_MAX = 0.5    # max reward per step
    SHIELD_D_MAX = 1.5    # distance at which reward decays to zero

    def _compute_shield_reward(self, agent, world):
        """
        Compute geometric shielding reward for Agent 2 in guard mode.

        For each fixed hunter, compute a blocking point on the hunter→prey
        line (biased toward prey side), then reward Agent 2 for proximity
        to that point. Multi-hunter aggregation uses max.

        Returns:
            float: shielding reward in [0, SHIELD_R_MAX]
        """
        prey = self.good_agents(world)[0]
        fixed_advs = self.fixed_adversaries(world)

        best_score = 0.0
        for adv in fixed_advs:
            # Blocking point: (1-alpha)*hunter + alpha*prey
            block_point = adv.state.p_pos + self.SHIELD_ALPHA * (
                prey.state.p_pos - adv.state.p_pos
            )
            dist_to_block = np.sqrt(np.sum(np.square(
                agent.state.p_pos - block_point
            )))
            score = self.SHIELD_R_MAX * max(0.0, 1.0 - dist_to_block / self.SHIELD_D_MAX)
            best_score = max(best_score, score)

        return best_score

    # ── Reward functions (Faction Re-alignment v3, Orthogonalized) ─────
    #
    # ALL agents' rewards depend on world.rule:
    #   reward = rule * r_hunt + (1 - rule) * r_guard
    #
    # Design principle: no event has opposite signs in r_hunt vs r_guard.
    # Each row is either same-sign (additive) or one-side-zero (orthogonal).
    #
    # Rule=1 (Full Hunt):  {0,1,2} hunt {3}
    # Rule=0 (Bodyguard):  {0,1} hunt {3}, {2} shields {3}, {3} seeks {2}

    def reward(self, agent, world):
        if getattr(agent, 'variable', False):
            return self._variable_agent_reward(agent, world)
        elif agent.adversary:
            return self._fixed_hunter_reward(agent, world)
        else:
            return self._prey_reward(agent, world)

    def _fixed_hunter_reward(self, agent, world):
        """
        Fixed Hunters (Agent 0, 1).

        Rule=1 (Hunt mode):
            - Shaped: -0.1 * dist_to_prey  (get closer)
            - Catch prey: +10
            - Collide with Agent 2 (teammate): -1.0  (don't block each other)

        Rule=0 (Bodyguard mode):
            - Shaped: -0.1 * dist_to_prey  (still chase prey)
            - Catch prey: +10
            - Collide with Agent 2 (obstacle): -0.5  (it's annoying, avoid it)

        Interpolation: rule * r_hunt + (1-rule) * r_guard
        """
        prey = self.good_agents(world)[0]
        variable = self.variable_agents(world)[0]
        dist_to_prey = self._dist(agent, prey)
        collide_prey = self.is_collision(agent, prey)
        collide_var = self.is_collision(agent, variable)

        # ── Rule=1 component: Full Hunt ──
        r_hunt = 0.0
        r_hunt -= 0.1 * dist_to_prey          # shaped: approach prey
        if collide_prey:
            r_hunt += 10.0                     # main goal: catch prey
        if collide_var:
            r_hunt -= 1.0                      # mild: don't bump teammate

        # ── Rule=0 component: Bodyguard mode ──
        r_guard = 0.0
        r_guard -= 0.1 * dist_to_prey         # still chase prey
        if collide_prey:
            r_guard += 10.0                    # still want to catch prey
        if collide_var:
            r_guard -= 0.5                     # Agent 2 is obstacle, annoying

        # ── Interpolation + boundary ──
        rew = world.rule * r_hunt + (1 - world.rule) * r_guard
        rew -= self._bound_penalty(agent, world)
        return rew

    def _variable_agent_reward(self, agent, world):
        """
        Variable Agent (Agent 2) — Orthogonalized v3.

        Rule=1 (Hunt mode):
            - Shaped: -0.1 * dist_to_prey  (approach prey)
            - Catch prey: +10
            - Collide with Hunters (teammates): -1.0

        Rule=0 (Guard mode):
            - Shaped: +shield_reward  (geometric blocking point guidance)
            - Prey caught by Hunter: -10  (protection failed)
            - No collision bonus (removed: was +1, conflicted with hunt -1)
            - No approach-prey shaped (removed: same direction as hunt)

        Orthogonality: every event has one-side-zero across r_hunt/r_guard.
        """
        prey = self.good_agents(world)[0]
        fixed_advs = self.fixed_adversaries(world)
        dist_to_prey = self._dist(agent, prey)

        # ── Rule=1 component: Hunt ──
        r_hunt = 0.0
        r_hunt -= 0.1 * dist_to_prey          # shaped: approach prey
        if self.is_collision(prey, agent):
            r_hunt += 10.0                     # catch prey
        for adv in fixed_advs:
            if self.is_collision(agent, adv):
                r_hunt -= 1.0                  # don't bump teammates

        # ── Rule=0 component: Guard (orthogonalized) ──
        r_guard = 0.0

        # Dense signal: geometric blocking point shielding reward
        r_guard += self._compute_shield_reward(agent, world)

        # Sparse signal: protection failure penalty
        for adv in fixed_advs:
            if self.is_collision(prey, adv):
                r_guard -= 10.0                # protection failed!

        # ── Interpolation + boundary ──
        rew = world.rule * r_hunt + (1 - world.rule) * r_guard
        rew -= self._bound_penalty(agent, world)
        return rew

    def _prey_reward(self, agent, world):
        """
        Prey (Agent 3) — Orthogonalized v3.

        Rule=1 (Hunt mode) — fear fixed hunters + Agent 2 collision:
            - Shaped: +0.1 * dist_to_each_fixed_hunter  (run from {0,1})
            - Caught by fixed hunter: -10
            - Caught by Agent 2: -10
            - No distance shaped for Agent 2 (removed: conflicted with guard proximity)

        Rule=0 (Guard mode) — fear fixed hunters, seek Agent 2:
            - Shaped: +0.1 * dist_to_each_fixed_hunter  (run from {0,1})
            - Caught by fixed hunter: -10
            - Proximity to Agent 2: +0.1 * max(0, safety_dist - dist)
            - No touching bonus (removed: conflicted with hunt collision -10)

        Orthogonality: fixed hunter distance/collision same-sign in both modes.
        Agent 2 interactions one-side-zero.
        """
        fixed_advs = self.fixed_adversaries(world)
        variable = self.variable_agents(world)[0]
        dist_to_var = self._dist(agent, variable)
        collide_var = self.is_collision(agent, variable)
        safety_dist = 2.0  # proximity shaping range

        # ── Rule=1 component: Fear fixed hunters + Agent 2 collision ──
        r_hunt = 0.0
        # Shaped: reward for distance from fixed hunters only
        for adv in fixed_advs:
            r_hunt += 0.1 * self._dist(agent, adv)
        # No Agent 2 distance shaped (removed for orthogonality)

        # Collision: caught by fixed hunter
        for adv in fixed_advs:
            if self.is_collision(agent, adv):
                r_hunt -= 10.0
        # Collision: caught by Agent 2 (still an enemy in hunt mode)
        if collide_var:
            r_hunt -= 10.0

        # ── Rule=0 component: Fear Hunters, Seek Agent 2 ──
        r_guard = 0.0
        # Shaped: reward for distance from fixed hunters (same as hunt — same sign)
        for adv in fixed_advs:
            r_guard += 0.1 * self._dist(agent, adv)

        # Collision: caught by fixed hunter (same sign as hunt)
        for adv in fixed_advs:
            if self.is_collision(agent, adv):
                r_guard -= 10.0

        # Seek Agent 2's protection (one-side-zero: only in guard)
        proximity = max(0.0, safety_dist - dist_to_var)
        r_guard += 0.1 * proximity

        # No touching bonus (removed: would conflict with hunt -10 collision)
        # No Agent 2 collision penalty in guard (one-side-zero)

        # ── Interpolation + boundary ──
        rew = world.rule * r_hunt + (1 - world.rule) * r_guard
        rew -= self._bound_penalty(agent, world)
        return rew

    # ── Observation ─────────────────────────────────────────────────────

    def observation(self, agent, world):
        """
        Observation for each agent (Rule is NOT included).

        Contains:
            - agent's own velocity (2,)
            - agent's own position (2,)
            - relative positions of landmarks (num_landmarks * 2,)
            - relative positions of other agents (num_other_agents * 2,)
            - velocities of prey agents (num_prey * 2,)
        """
        # Landmark relative positions
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # Other agents' relative positions and prey velocities
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel
        )

    # ── Benchmark data ──────────────────────────────────────────────────

    def benchmark_data(self, agent, world):
        collisions = 0
        if agent.adversary:
            for prey in self.good_agents(world):
                if self.is_collision(prey, agent):
                    collisions += 1
        return collisions
