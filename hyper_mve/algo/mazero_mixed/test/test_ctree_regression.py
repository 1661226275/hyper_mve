"""Stage-2 gate: vectorized ctree must reproduce scalar-baseline search behavior
in the all-cooperative special case (joint selection mode).

Usage:
  python test/test_ctree_regression.py capture   # BEFORE vectorization: writes golden JSON
  python test/test_ctree_regression.py check     # AFTER: compares against golden JSON

The scripted search drives cytree.Tree_batch directly with deterministic
synthetic inputs (fixed RNG), no torch/model involved. Captured signals:
per-simulation selection results (idx, actions) and final root values /
marginal visit counts / sampled visit counts.

After vectorization, `check` broadcasts scalar rewards/values to per-agent
vectors (identical entries = all-cooperative case) and runs select_mode=0
(joint). Every captured signal must match exactly; per-agent root values must
also all equal the golden scalar root value.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from core.mcts.ctree.ctree_sampled import cytree  # noqa: E402

GOLDEN = os.path.join(HERE, "ctree_golden.json")

ROOT_NUM = 4
N_AGENTS = 2
A = 6
SAMPLED = 5
SIMS = 25
SEED = 7
DISCOUNT = 0.997
PB_C_BASE, PB_C_INIT = 19652.0, 1.25
RHO, LAM = 0.75, 0.8
DELTA_LB = 0.01


def synth(rng, sim):
    """Deterministic synthetic leaf evaluations for simulation index `sim`."""
    rewards = rng.uniform(-1, 1, ROOT_NUM).astype(np.float32)
    values = rng.uniform(-2, 2, ROOT_NUM).astype(np.float32)
    logits = rng.uniform(0, 1, (ROOT_NUM, N_AGENTS, A)).astype(np.float32)
    probs = logits / logits.sum(-1, keepdims=True)
    beta = probs.copy()
    return rewards, values, probs, beta


def run(vectorized: bool):
    rng = np.random.RandomState(SEED)
    kwargs = {}
    try:
        trees = cytree.Tree_batch(
            ROOT_NUM, N_AGENTS, A, SAMPLED, SIMS, DELTA_LB, SEED, RHO, LAM, 0
        )
        has_mode = True
    except TypeError:
        trees = cytree.Tree_batch(
            ROOT_NUM, N_AGENTS, A, SAMPLED, SIMS, DELTA_LB, SEED, RHO, LAM
        )
        has_mode = False

    def widen(x):
        # scalar per root -> per-agent vector with identical entries
        if vectorized:
            return np.repeat(x.reshape(ROOT_NUM, 1), N_AGENTS, axis=1).astype(np.float32)
        return x

    rewards, values, probs, beta = synth(rng, -1)
    noises = rng.dirichlet([0.3] * A, ROOT_NUM * N_AGENTS).astype(np.float32).reshape(
        ROOT_NUM, N_AGENTS, A
    )
    trees.prepare(widen(rewards * 0.0), widen(values), probs, beta, SAMPLED, 0.0, noises)

    trace = []
    for sim in range(SIMS):
        idx, idy, acts = trees.batch_selection(PB_C_BASE, PB_C_INIT, DISCOUNT)
        trace.append(
            {"idx": list(map(int, idx)), "acts": np.asarray(acts).astype(int).tolist()}
        )
        rewards, values, probs, beta = synth(rng, sim)
        trees.batch_expansion_and_backup(
            sim + 1, DISCOUNT, SAMPLED, widen(rewards), widen(values), probs, beta
        )

    out = {
        "trace": trace,
        "root_values": np.asarray(trees.get_roots_values()).reshape(-1).round(5).tolist(),
        "marginal_visits": np.asarray(trees.get_roots_marginal_visit_count())
        .astype(int)
        .tolist(),
        "sampled_visits": [v.astype(int).tolist() for v in trees.get_roots_sampled_visit_count()],
        "sampled_qvalues": [
            np.asarray(q).reshape(-1).round(4).tolist()
            for q in trees.get_roots_sampled_qvalues(DISCOUNT)
        ],
    }
    if vectorized and hasattr(trees, "get_roots_values_vec"):
        out["root_values_vec"] = (
            np.asarray(trees.get_roots_values_vec()).reshape(ROOT_NUM, N_AGENTS).round(5).tolist()
        )
    return out, has_mode


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "capture":
        out, has_mode = run(vectorized=False)
        assert not has_mode, "capture must run against the ORIGINAL (pre-vectorization) build"
        with open(GOLDEN, "w") as f:
            json.dump(out, f)
        print(f"golden captured -> {GOLDEN}")
    else:
        with open(GOLDEN) as f:
            golden = json.load(f)
        out, has_mode = run(vectorized=True)
        assert has_mode, "check must run against the vectorized build (select_mode arg)"
        errs = []
        for k in ("trace", "marginal_visits", "sampled_visits"):
            if out[k] != golden[k]:
                errs.append(k)
        for k in ("root_values", "sampled_qvalues"):
            g = np.array(
                [x for row in golden[k] for x in (row if isinstance(row, list) else [row])]
            )
            o = np.array(
                [x for row in out[k] for x in (row if isinstance(row, list) else [row])]
            )
            if g.shape != o.shape or not np.allclose(g, o, atol=2e-3):
                errs.append(k)
        if "root_values_vec" in out:
            rv = np.array(out["root_values_vec"])
            if not np.allclose(rv, rv[:, [0]], atol=1e-5):
                errs.append("per-agent root values differ in coop case")
            if not np.allclose(rv[:, 0], np.array(golden["root_values"]), atol=2e-3):
                errs.append("per-agent root value != golden scalar value")
        if errs:
            print("REGRESSION FAIL:", errs)
            sys.exit(1)
        print("REGRESSION PASS: vectorized joint mode reproduces scalar baseline")
