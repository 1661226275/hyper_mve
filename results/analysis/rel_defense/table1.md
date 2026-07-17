# Table 1 — baseline comparison summary (rel_duo, N=2, |G|=5)

| Method | Cell | Seeds | Env-step budget | Return (mean±SEM) | Δ vs model-free | Seen regimes | Held-out regimes | Zero-shot gap | NashConv (mean, LB) | Walltime/seed |
|---|---|---|---|---|---|---|---|---|---|---|
| DR-MBHGL (ours) | rel_gate_duo | 1 | 2,000,000 | 13.92 (single seed) | -20.69 | — | — | — | 4.58 | — |
| MAPPO | rel_gate_duo | 1 | 2,000,000 | 34.61 (single seed) | 0 (ref) | — | — | — | 0.00 | 1.0 h |
| DR-MBHGL (ours) | rel_zero_shot_duo | 1 | 2,000,000 | 16.39 (single seed) | -17.70 | 13.55 | 20.64 | +7.08 | — | — |
| MAMBA | rel_zero_shot_duo | 1 (proj.) | 2,000,000 * | 15.90 * | -18.19 * | 18.03 * | 12.57 * | -5.46 * | — | in progress |
| MAPPO | rel_zero_shot_duo | 1 | 2,000,000 | 34.09 (single seed) | 0 (ref) | 41.79 | 22.53 | -19.26 | — | 1.0 h |

Notes: all values are 30 deterministic episodes/regime evaluated at each method's 2M-env-step checkpoint (one uniform protocol; `hyper` = step_20000.pt — 20k gradient steps × 100 env steps/step). `Δ vs model-free`: model-free methods interact with the real environment directly and carry no world-model error, so at a matched env-step budget in a deterministic training scenario their return is the natural upper reference; a model-based method's gap to it reflects how faithfully its world model simulates the environment (smaller |Δ| = higher fidelity). `Zero-shot gap` = unseen − seen: positive means training experience TRANSFERS to regimes never seen in training (knowledge transfer), negative means performance collapses off the training distribution — the primary generalization criterion here, ahead of absolute return. NashConv values are lower bounds (approximate DQN best response, budget in game_metrics JSON). Single-seed rows carry no error estimate.
`*` / `(proj.)` = fitted saturating-curve extrapolation of MAMBA's accurate 30-episode evals at intermediate checkpoints to the full 2M env-step budget — provisional, replaced by the measured eval_report once training completes.