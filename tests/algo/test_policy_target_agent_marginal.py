"""The action-axis (per-agent marginal) policy targets.

Two things are pinned here, and the first matters more than the target itself.

**The no-op result.** Because the policy head is factorized
(``sampled_actions_log_prob = per_agent_log_prob.sum(dim=1)``), the ``visit``
loss ``-sum_c pi(c) sum_i log p_i(a_i^c)`` reassociates into
``-sum_i sum_a [sum_{c: a_i^c=a} pi(c)] log p_i(a)`` — and that bracket is
*exactly* what the C++ tree's ``get_marginal_visit_count`` computes
(``cnode.cpp:95-105``). So feeding ``marginal_visit_count`` to the loss as a
"per-agent visit target" would be mathematically identical to the ``visit``
target already shipped. ``test_visit_prior_at_huge_temperature_is_the_visit_loss``
is that statement as an executable proof, so the proposal is not re-litigated.

The corollary is the design constraint: the sampled-child axis is not an
independent design space, it is a parameterization of the ``(N, A)`` marginal.
Every target differs only in how it aggregates children onto actions.

**What ``agent_q_softmax`` actually changes** is therefore narrow and precise:
it aggregates by a visit-weighted average INSIDE the exponent instead of a sum
outside it, which removes a MULTIPLICITY term. ``q_softmax`` gives an action
carried by ``m`` children ``m`` exp-terms; this gives it one, estimated from all
``m``. Tests 3a/3b separate the two halves of that claim — 3a holds the
advantages equal so only multiplicity varies, 3b varies the advantages so only
the aggregation operator can explain the difference. Without 3b,
``agent_q_softmax`` would be indistinguishable from a plain multiplicity
correction to ``q_softmax``.

Why multiplicity is not a corner case: under ``--root_cover star`` the cover
pins each agent at one draw from its own noised prior across ``A`` of the ~``2A``
children (``cnode.cpp:341-365``), so in the flat-advantage limit ``q_softmax``
puts ~50% of that agent's target mass on a single prior sample.
"""
from __future__ import annotations

import os
import sys

import pytest

# the vendored fork imports its own package as top-level ``core`` (see
# core/train.py), so it has to be on sys.path -- same shim as
# tests/algo/test_policy_target_blend.py.
_FORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hyper_mve", "algo", "mazero_mixed",
)
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

torch = pytest.importorskip("torch")

from core.train import agent_marginal_target, policy_loss_step  # noqa: E402

BATCH, C, N, A = 4, 5, 2, 6
SIMS = 25.0


class _Cfg:
    def __init__(self, policy_target_type, policy_target_temperature=1.0,
                 action_space_size=A):
        self.policy_target_type = policy_target_type
        self.policy_target_temperature = policy_target_temperature
        self.action_space_size = action_space_size


@pytest.fixture
def batch():
    """A batch that deliberately contains action COLLISIONS and a masked row.

    Collisions are the whole point: an injective child->action map makes
    ``agent_q_softmax`` and ``q_softmax`` identical (pinned separately), so a
    fixture without them would pass every test for the wrong reason. Row 3 is
    fully masked -- the out-of-trajectory case the -1e4 sentinel exists for.
    """
    g = torch.Generator().manual_seed(0)
    adv = torch.randn(BATCH, C, N, generator=g)
    adv = adv - adv.mean(dim=1, keepdim=True)          # centered, like the real thing
    mask = torch.ones(BATCH, C)
    mask[3] = 0.0
    visits = torch.tensor([
        [10.0, 6.0, 4.0, 3.0, 2.0],
        [1.0, 1.0, 20.0, 2.0, 1.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],                     # masked row
    ])
    visit_policy = visits / SIMS
    actions = torch.tensor([
        # agent 0 repeats action 1 on three children; agent 1 repeats action 4
        [[1, 4], [1, 0], [1, 4], [2, 3], [5, 4]],
        [[0, 0], [3, 3], [0, 1], [3, 0], [2, 2]],
        [[2, 5], [2, 5], [4, 1], [4, 1], [0, 3]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],      # masked row: real indices, zero weight
    ])
    # masked slots carry adv == 0 exactly, as reanalyze_worker.py guarantees
    adv = adv * mask.unsqueeze(-1)
    log_prob_full = torch.log_softmax(torch.randn(BATCH, N, A, generator=g), dim=-1)
    return actions, visit_policy, adv, mask, log_prob_full


def _per_agent_log_prob(log_prob_full, actions):
    """The (batch, N, C) gather the sampled-child targets consume."""
    return log_prob_full.gather(dim=2, index=actions.transpose(1, 2).long())


def _loss(cfg, actions, visit_policy, adv, mask, log_prob_full):
    palp = _per_agent_log_prob(log_prob_full, actions)
    return policy_loss_step(cfg, palp, palp.sum(dim=1), visit_policy, adv, mask,
                            log_prob_full, actions)


# --------------------------------------------------------------------------
# 1. the no-op result, as an executable proof
# --------------------------------------------------------------------------

def test_visit_prior_at_huge_temperature_is_the_visit_loss(batch):
    """``agent_q_blend`` at tau -> inf IS the visit target, exactly.

    This is the marginal-visit no-op in executable form. The advantage term
    vanishes, leaving softmax(log n_i(a)) = the normalized marginal visit
    count -- i.e. precisely ``get_marginal_visit_count`` -- and the resulting
    loss must equal the joint-visit loss. Run on a fixture WITH collisions and
    padding, since those are the only places the two could have differed.
    """
    actions, visit_policy, adv, mask, lpf = batch
    visit_loss = _loss(_Cfg("visit"), actions, visit_policy, adv, mask, lpf)
    agent_loss = _loss(_Cfg("agent_q_blend", 1e6), actions, visit_policy, adv, mask, lpf)

    # the visit branch does not renormalize; row sums are visits/SIMS, so
    # compare against the same rows renormalized to 1.
    live = mask.sum(dim=1) > 0
    row_sum = (visit_policy * mask).sum(dim=1)
    scaled = visit_loss.clone()
    scaled[live] = visit_loss[live] / row_sum[live]

    torch.testing.assert_close(agent_loss[live], scaled[live], atol=1e-4, rtol=1e-4)
    assert agent_loss[~live].abs().max() == 0.0


def test_marginal_visit_count_is_reproduced_exactly(batch):
    """The n_i(a) accumulator equals the C++ marginalisation, per definition.

    ``logits(j, children_action[i][j]) += child->visit_count`` -- pinned here
    against an independent loop so the equivalence above cannot rot silently.
    """
    actions, visit_policy, adv, mask, _ = batch
    # use_visit_prior=True is what puts n_i(a) in the logits at all; without it
    # a huge temperature gives the UNIFORM target, not the visit marginal.
    T = agent_marginal_target(actions, visit_policy, adv, mask, A, 1e6,
                              use_visit_prior=True)

    expected = torch.zeros(BATCH, N, A)
    for b in range(BATCH):
        for c in range(C):
            if mask[b, c] == 0:
                continue
            for i in range(N):
                expected[b, i, int(actions[b, c, i])] += visit_policy[b, c]
    denom = expected.sum(dim=2, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(T, expected / denom, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 2. injectivity: the exact condition under which the new target is old news
# --------------------------------------------------------------------------

def test_injective_child_action_map_recovers_q_softmax(batch):
    """With no collisions, agent_q_softmax IS q_softmax -- so collisions are
    the *only* thing separating them."""
    _, visit_policy, adv, mask, lpf = batch
    # one distinct action per child for both agents
    actions = torch.stack([torch.stack([torch.tensor([c, C - 1 - c])
                                        for c in range(C)])] * BATCH)
    agent_loss = _loss(_Cfg("agent_q_softmax"), actions, visit_policy, adv, mask, lpf)
    q_loss = _loss(_Cfg("q_softmax"), actions, visit_policy, adv, mask, lpf)
    torch.testing.assert_close(agent_loss, q_loss, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# 3a. multiplicity invariance, advantages held EQUAL
# --------------------------------------------------------------------------

def test_splitting_a_child_with_equal_advantage_leaves_agent_target_fixed():
    """Split one child into two same-action children carrying the IDENTICAL
    advantage and half the visits each.

    ``agent_q_softmax`` must be bit-unchanged: n_i(a) re-sums to the same
    total and qbar_i(a) averages equal values. ``q_softmax`` must NOT be --
    it gains a second exp-term and doubles that action's weight. Nothing but
    multiplicity varies, so this isolates the star mechanism.
    """
    adv_vals, tau = [0.8, -0.3, -0.5], 1.0

    def build(split):
        """split=False: 3 children. split=True: child 0 duplicated, visits halved."""
        if split:
            acts = torch.tensor([[[0, 0], [0, 0], [1, 1], [2, 2]]])
            vis = torch.tensor([[3.0, 3.0, 6.0, 6.0]]) / SIMS
            adv = torch.tensor([[[adv_vals[0]] * N, [adv_vals[0]] * N,
                                 [adv_vals[1]] * N, [adv_vals[2]] * N]])
        else:
            acts = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
            vis = torch.tensor([[6.0, 6.0, 6.0]]) / SIMS
            adv = torch.tensor([[[adv_vals[0]] * N, [adv_vals[1]] * N,
                                 [adv_vals[2]] * N]])
        return acts, vis, adv, torch.ones(1, acts.shape[1])

    a0, v0, d0, m0 = build(False)
    a1, v1, d1, m1 = build(True)
    T0 = agent_marginal_target(a0, v0, d0, m0, A, tau)
    T1 = agent_marginal_target(a1, v1, d1, m1, A, tau)
    torch.testing.assert_close(T0, T1, atol=1e-6, rtol=1e-6)

    # ...and the sampled-child target does move, by the multiplicity factor.
    from core.train import policy_target_weights
    w0 = policy_target_weights(d0, m0, tau)
    w1 = policy_target_weights(d1, m1, tau)
    mass0 = float(w0[0, 0, 0])               # action 0 sits on child 0 only
    mass1 = float(w1[0, 0, 0] + w1[0, 1, 0])  # now spread over children 0 and 1
    # compare ODDS, not probability: the weights are softmax-normalized, so a
    # term that is already 62% of the mass cannot double to 124%. The
    # multiplicity factor shows up exactly in mass/(1-mass).
    odds0, odds1 = mass0 / (1 - mass0), mass1 / (1 - mass1)
    assert abs(odds1 / odds0 - 2.0) < 1e-4, (
        f"q_softmax should double action 0's ODDS on the split "
        f"(x{odds1 / odds0:.4f}, mass {mass0:.4f} -> {mass1:.4f}); if it does "
        f"not, multiplicity is gone and 3a no longer isolates anything")


# --------------------------------------------------------------------------
# 3b. averaging vs sum-exp, advantages DIFFERENT
# --------------------------------------------------------------------------

def test_split_with_different_advantages_averages_rather_than_sums_exp():
    """Same split, but the two same-action children carry DIFFERENT advantages.

    ``agent_q_softmax`` must equal exp(visit-weighted MEAN / tau); ``q_softmax``
    forms sum of exp(adv/tau). The two disagree on the aggregation OPERATOR,
    not merely on multiplicity -- which is what stops this target from being
    just a multiplicity correction. By Jensen the sum-exp form is biased
    upward on a spread group; pin that the new target is not.
    """
    tau = 1.0
    hi, lo, other = 1.2, -0.4, -0.8
    acts = torch.tensor([[[0, 0], [0, 0], [1, 1]]])
    vis = torch.tensor([[4.0, 12.0, 9.0]]) / SIMS          # deliberately UNEQUAL
    adv = torch.tensor([[[hi] * N, [lo] * N, [other] * N]])
    mask = torch.ones(1, 3)

    T = agent_marginal_target(acts, vis, adv, mask, A, tau)

    # the exact visit-weighted mean, computed independently
    qbar0 = (4.0 * hi + 12.0 * lo) / 16.0
    ref = torch.zeros(A)
    ref[0] = torch.tensor(qbar0 / tau).exp()
    ref[1] = torch.tensor(other / tau).exp()
    ref = ref / ref.sum()
    torch.testing.assert_close(T[0, 0], ref, atol=1e-6, rtol=1e-6)

    # the group is spread, so Jensen bites: sum-exp puts strictly more mass on
    # action 0 than exp-of-mean does, over and above the multiplicity factor.
    from core.train import policy_target_weights
    w = policy_target_weights(adv, mask, tau)
    q_mass0 = float(w[0, 0, 0] + w[0, 1, 0])
    assert q_mass0 > float(T[0, 0, 0]), (
        "sum-exp should exceed exp-of-mean on a spread group")
    # and the gap is NOT the constant factor 3a would predict from multiplicity
    # alone: with equal advantages the ratio would be exactly 2 here.
    equal_adv = torch.tensor([[[qbar0] * N, [qbar0] * N, [other] * N]])
    w_eq = policy_target_weights(equal_adv, mask, tau)
    eq_mass0 = float(w_eq[0, 0, 0] + w_eq[0, 1, 0])
    assert abs(q_mass0 - eq_mass0) > 1e-3, (
        f"spread vs equal advantages must give different q_softmax mass "
        f"({q_mass0:.5f} vs {eq_mass0:.5f}); if equal, the operator "
        f"difference is invisible and only multiplicity is being tested")


# --------------------------------------------------------------------------
# 4-6. numerical safety
# --------------------------------------------------------------------------

def test_fully_masked_row_is_exactly_zero_and_never_nan(batch):
    actions, visit_policy, adv, mask, lpf = batch
    for tt in ("agent_q_softmax", "agent_q_blend"):
        loss = _loss(_Cfg(tt), actions, visit_policy, adv, mask, lpf)
        assert torch.isfinite(loss).all(), f"{tt} produced non-finite loss"
        assert loss[3] == 0.0, f"{tt} masked row must contribute exactly 0"


def test_masked_row_stays_zero_under_autocast(batch):
    """The -1e4 sentinel exists because fp16 turns -1e9/-inf into NaN, and
    `0 * NaN` survives the mask and poisons total_loss."""
    if not torch.cuda.is_available():
        pytest.skip("autocast fp16 path needs CUDA")
    actions, visit_policy, adv, mask, lpf = [t.cuda() for t in batch]
    with torch.cuda.amp.autocast():
        loss = _loss(_Cfg("agent_q_softmax"), actions, visit_policy, adv, mask, lpf)
    assert torch.isfinite(loss).all()
    assert float(loss[3]) == 0.0


def test_zero_visit_child_is_dropped_which_is_what_makes_the_endpoint_exact():
    """A 0-visit child carries no weight in a visit-weighted mean, so its
    action leaves ``present`` unless another child supplies it.

    This is deliberate, not an oversight: it is exactly why the
    ``temperature -> inf`` endpoint reproduces the ``visit`` target, which also
    weights such a child 0. ``q_softmax`` by contrast still gives it an
    exp-term -- so "agent_q_softmax == q_softmax under injectivity" is a claim
    about the VISITED children. Empty in practice at the root (the tree forces
    one visit per root child while num_simulations >= num_children), but it
    would silently change the endpoint if it ever regressed.
    """
    acts = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
    vis = torch.tensor([[10.0, 15.0, 0.0]]) / SIMS      # child 2 unvisited
    adv = torch.tensor([[[0.5] * N, [-0.2] * N, [3.0] * N]])
    mask = torch.ones(1, 3)

    T = agent_marginal_target(acts, vis, adv, mask, A, 1.0)
    assert T[0, 0, 2] == 0.0, "an unvisited child's action must carry no mass"
    torch.testing.assert_close(T[0, 0].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    # q_softmax does include it, and prominently -- its adv is the largest.
    from core.train import policy_target_weights
    w = policy_target_weights(adv, mask, 1.0)
    assert float(w[0, 2, 0]) > 0.5, (
        "q_softmax should weight the unvisited child heavily here; if not, "
        "this test no longer demonstrates the difference")


def test_absent_actions_get_exactly_zero_mass_and_rows_normalize(batch):
    actions, visit_policy, adv, mask, _ = batch
    for use_prior in (False, True):
        T = agent_marginal_target(actions, visit_policy, adv, mask, A, 1.0,
                                  use_visit_prior=use_prior)
        assert (T >= 0).all(), "a negative target weight flips the loss sign"
        for b in range(BATCH):
            for i in range(N):
                present = {int(actions[b, c, i]) for c in range(C) if mask[b, c] > 0}
                for a in range(A):
                    if a not in present:
                        assert T[b, i, a] == 0.0, (
                            f"action {a} is on no live child but carries mass")
        live = mask.sum(dim=1) > 0
        torch.testing.assert_close(
            T[live].sum(dim=2), torch.ones(int(live.sum()), N), atol=1e-5, rtol=1e-5)
        assert T[~live].sum() == 0.0
