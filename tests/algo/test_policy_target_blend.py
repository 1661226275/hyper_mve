"""``visit_q_blend`` must actually interpolate between the two shipped targets.

The blend exists because ``q_softmax`` throws the visit allocation away — it
weights a 1-visit Q identically to a 20-visit Q, and at ``root_cover=star`` the
budget is only ~1.9 visits per child. Restoring visits as a multiplicative
prior is only defensible if the two endpoints are the arms we already have data
for; otherwise it is a third unrelated target and none of the 2x2 comparisons
transfer.

The endpoint that carries the most weight is ``temperature -> inf``: it must
reproduce the visit target at the LOSS level, not merely in the weights. That
holds because ``sampled_actions_log_prob = per_agent_log_prob.sum(dim=1)``
(``core/train.py``) and ``visit(c)`` does not depend on the agent, so the joint
form and the per-agent form are the same sum reassociated. If that ever stops
being true, the blend silently stops being a superset of the visit target.

Also pinned here: the sign trap that makes the naive ``visit(c) * adv_i(c)``
product unusable. ``adv`` is centered, so ~half the children are negative, and a
negative weight in ``-sum w log p`` becomes an unbounded incentive to drive
``p -> 0``. The blend must never produce a negative weight.
"""
from __future__ import annotations

import os
import sys

import pytest

# the vendored fork imports its own package as top-level ``core`` (see
# core/train.py:14), so it has to be on sys.path -- same shim as
# tests/algo/test_act_fn_contract.py.
_FORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hyper_mve", "algo", "mazero_mixed",
)
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

torch = pytest.importorskip("torch")

from core.train import policy_loss_step, policy_target_weights   # noqa: E402

BATCH, C, N = 4, 5, 2


class _Cfg:
    def __init__(self, policy_target_type, policy_target_temperature=1.0):
        self.policy_target_type = policy_target_type
        self.policy_target_temperature = policy_target_temperature


@pytest.fixture
def batch():
    """A batch with an uneven visit allocation and centered advantages.

    Row 3 is fully masked — the out-of-trajectory case that made the -1e4
    masking sentinel necessary in the first place.
    """
    g = torch.Generator().manual_seed(0)
    adv = torch.randn(BATCH, C, N, generator=g)
    adv = adv - adv.mean(dim=1, keepdim=True)              # centered, like the real thing
    mask = torch.ones(BATCH, C)
    mask[3] = 0.0
    visits = torch.tensor([
        [10.0, 6.0, 4.0, 3.0, 2.0],                        # uneven
        [1.0, 1.0, 20.0, 2.0, 1.0],                        # one dominant child
        [5.0, 5.0, 5.0, 5.0, 5.0],                         # uniform
        [0.0, 0.0, 0.0, 0.0, 0.0],                         # masked row
    ])
    visit_policy = visits / 25.0                           # num_simulations
    per_agent_log_prob = torch.log_softmax(
        torch.randn(BATCH, N, C, generator=g), dim=-1)
    return adv, mask, visit_policy, per_agent_log_prob


def test_weights_are_a_distribution_and_never_negative(batch):
    """The whole reason the visit term goes in the exponent."""
    adv, mask, visit_policy, _ = batch
    w = policy_target_weights(adv, mask, 1.0, visit_policy)
    assert torch.all(w >= 0.0), "a negative target weight is unbounded gradient ascent"
    assert torch.isfinite(w).all()
    live = w[:3].sum(dim=1)                                # (batch-1, N)
    torch.testing.assert_close(live, torch.ones_like(live))
    # the fully-masked row stays exactly 0, not 0/0
    assert torch.all(w[3] == 0.0)


def test_uniform_visits_recover_q_softmax_exactly(batch):
    """Endpoint 1: a constant log-visit term drops out of the softmax."""
    adv, mask, visit_policy, _ = batch
    uniform = torch.full_like(visit_policy, 1.0 / C)
    blended = policy_target_weights(adv, mask, 1.0, uniform)
    pure_q = policy_target_weights(adv, mask, 1.0, None)
    torch.testing.assert_close(blended, pure_q)


def test_large_temperature_recovers_the_visit_target_weights(batch):
    """Endpoint 2, at the weight level: adv/tau vanishes, leaving softmax(log v)."""
    adv, mask, visit_policy, _ = batch
    w = policy_target_weights(adv, mask, 1e6, visit_policy)
    want = (visit_policy * mask) / (visit_policy * mask).sum(
        dim=1, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(w[:3], want[:3].unsqueeze(-1).expand(-1, -1, N),
                               atol=1e-5, rtol=1e-4)


def test_large_temperature_recovers_the_visit_target_LOSS(batch):
    """Endpoint 2, at the loss level — the load-bearing one.

    Uses a visit target that already sums to 1 per row, so the two paths differ
    only in factorization and not in an overall scale. The real
    ``target_sampled_policies`` is ``visit_count / num_simulations``, whose row
    sum is 1 only when the visit counts sum to exactly ``num_simulations``; that
    invariant is asserted nowhere (it is commented out in
    ``mcts_sampled.py``), which is why this test normalizes explicitly rather
    than assuming it.
    """
    adv, mask, visit_policy, per_agent_log_prob = batch
    normed = (visit_policy * mask) / (visit_policy * mask).sum(
        dim=1, keepdim=True).clamp_min(1e-8)
    joint_log_prob = per_agent_log_prob.sum(dim=1)               # (batch, C)

    visit_loss = policy_loss_step(
        _Cfg("visit"), per_agent_log_prob, joint_log_prob, normed, adv, mask)
    blend_loss = policy_loss_step(
        _Cfg("visit_q_blend", 1e6), per_agent_log_prob, joint_log_prob,
        normed, adv, mask)

    torch.testing.assert_close(blend_loss[:3], visit_loss[:3],
                               atol=1e-4, rtol=1e-4)


def test_blend_differs_from_both_endpoints_at_a_working_temperature(batch):
    """Guards against the blend silently degenerating into one of its parents —
    which would make a whole sweep look like a null result."""
    adv, mask, visit_policy, per_agent_log_prob = batch
    joint_log_prob = per_agent_log_prob.sum(dim=1)
    normed = (visit_policy * mask) / (visit_policy * mask).sum(
        dim=1, keepdim=True).clamp_min(1e-8)

    losses = {
        t: policy_loss_step(_Cfg(t, 1.0), per_agent_log_prob, joint_log_prob,
                            normed, adv, mask)[:3]
        for t in ("visit", "q_softmax", "visit_q_blend")
    }
    assert not torch.allclose(losses["visit_q_blend"], losses["visit"])
    assert not torch.allclose(losses["visit_q_blend"], losses["q_softmax"])


def test_temperature_monotonically_moves_between_the_endpoints(batch):
    """Raising tau moves the target toward visit and away from q_softmax.

    Without this, "tau" could be an arbitrary reparametrization and a sweep over
    it would not be a sweep over anything interpretable.

    The q_softmax reference is taken at the SAME tau, which is the only
    comparison that means anything: the log-visit term has a fixed coefficient,
    so shrinking tau inflates ``adv/tau`` against it and the blend approaches
    ``softmax(adv/tau)`` — q_softmax at *that* temperature, not at a fixed 1.0.
    Comparing against a fixed-temperature q_softmax conflates "how much visit
    is mixed in" with "how sharp the advantage term is" and is non-monotone.
    """
    adv, mask, visit_policy, _ = batch
    visit_w = policy_target_weights(adv, mask, 1e6, visit_policy)[:3]

    prev_to_visit, prev_to_q = None, None
    for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
        w = policy_target_weights(adv, mask, tau, visit_policy)[:3]
        d_visit = (w - visit_w).abs().sum().item()
        d_q = (w - policy_target_weights(adv, mask, tau, None)[:3]).abs().sum().item()
        if prev_to_visit is not None:
            assert d_visit < prev_to_visit, f"tau={tau} moved AWAY from visit"
            assert d_q > prev_to_q, f"tau={tau} moved TOWARD q_softmax(tau)"
        prev_to_visit, prev_to_q = d_visit, d_q


def test_zero_visit_child_is_penalized_but_finite(batch):
    """log(0) must not become NaN and must not poison the row."""
    adv, mask, visit_policy, _ = batch
    vp = visit_policy.clone()
    vp[0, 4] = 0.0                                          # unvisited, still unmasked
    w = policy_target_weights(adv, mask, 1.0, vp)
    assert torch.isfinite(w).all()
    assert w[0, 4].max().item() < 1e-6                      # effectively excluded
    torch.testing.assert_close(w[0].sum(dim=0), torch.ones(N))
