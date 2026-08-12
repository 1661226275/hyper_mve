#!/usr/bin/env python
"""Generate the seed-1 / seed-2 replication queue for the v7 500k wave.

Written as a generator rather than a hand-typed JSON because it is 30 jobs and
the ordering carries the priority: seed 1 is emitted in full before seed 2 (the
user asked for the seeds sequentially), and within a seed the jobs are ordered
by how much they matter to the headline, since the tail is what gets dropped if
the wave is cut short.

Order within a seed:
  1. method + Module-1 ablation        -- the headline pair
  2. baselines on the full regime set  -- the comparison table
  3. generalization (holdout partition)
  4. search-module arms                -- the ablation table's second half
  5. margvisit                         -- a NULL CONTROL (provably the same
     estimator as `visit`); it measures the seed-noise floor, so it is genuinely
     informative at n>1, but it is last because it cannot change a ranking.

Usage::

    python scripts/grids/make_seed_queue.py > scripts/grids/v7_seed12_queue.json
"""
from __future__ import annotations

import json

OUT = "results_v7_500k"
STEPS = 500_000
EPISODES = 128
METHOD = "ref_bc_anneal_scaled_hardval_decoupled"

MAZERO_ARMS = [
    ("method", METHOD, "relation_coopmix"),
    ("abl_module1_no_subjective", "ref_bc_anneal_scaled_no_subjective_decoupled",
     "relation_coopmix"),
]
BASELINES = ["mamba", "mbom", "m3w_adapted", "happo"]
HOLDOUT_ALGOS = ["mamba", "mbom", "m3w_adapted", "happo"]
SEARCH_ARMS = [
    ("abl_search_cover", METHOD + "_cover"),
    ("abl_search_qtarget", METHOD + "_qtarget"),
    ("abl_search_mctsfix", METHOD + "_mctsfix"),
]


def mazero(name, arm, env, seed):
    return {
        "name": f"{name}_s{seed}", "algo": "mazero_mixed", "ablation": arm,
        "env": env, "seed": seed, "total_env_steps": STEPS,
        "episodes": EPISODES, "num_pmcts": 16, "env_steps_per_grad": 16,
        "out": OUT,
    }


def baseline(algo, env, seed, tag):
    return {
        "name": f"{tag}_{algo}_s{seed}", "algo": algo, "env": env, "seed": seed,
        "total_env_steps": STEPS, "episodes": EPISODES, "out": OUT,
    }


def main() -> int:
    jobs = []
    for seed in (1, 2):
        for name, arm, env in MAZERO_ARMS:                 # 1. headline pair
            jobs.append(mazero(name, arm, env, seed))
        for algo in BASELINES:                             # 2. comparison
            jobs.append(baseline(algo, "relation_coopmix", seed, "base"))
        jobs.append(mazero("gen_method_holdout", METHOD,   # 3. generalization
                           "relation_coopmix_holdout", seed))
        for algo in HOLDOUT_ALGOS:
            jobs.append(baseline(algo, "relation_coopmix_holdout", seed, "gen"))
        for name, arm in SEARCH_ARMS:                      # 4. search ablation
            jobs.append(mazero(name, arm, "relation_coopmix", seed))
        jobs.append(mazero("abl_search_margvisit",         # 5. null control
                           METHOD + "_margvisit", "relation_coopmix", seed))
    print(json.dumps(jobs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
