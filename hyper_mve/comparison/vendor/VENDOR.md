# Vendored upstream clones

Pristine official clones (each with its own `.git`, untracked by the parent
repo — this manifest pins provenance). **Source-integrity rule: only
adaptive env/metric-integration edits are permitted inside these trees**
(enumerated per baseline in the phase-2 realignment plan); everything else
stays byte-identical to the pinned SHA. `git -C <dir> diff` shows the full
local delta at any time.

| Dir | Upstream | Pinned SHA (2026-07-17) | Role |
|---|---|---|---|
| `HARL/` | https://github.com/PKU-MARL/HARL.git | `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` | HAPPO baseline (per-agent rewards via `state_type="FP"`); HASAC reference for M3W-adapted SAC base |
| `MBOM/` | https://github.com/PKU-RL/MBOM.git | `d79ba71af843a9073bce0501a0cd85230585fdc0` | MBOM / MBOM-oracle baseline (2-player, RelationCommons duo only) |
| `m3w-marl/` | https://github.com/zhaozijie2022/m3w-marl.git | `80c373fb4a0cf2a5f8a4515086151fc306f37f3f` | MoE world-model modules imported unmodified by `comparison/m3w_adapted/` |
| `mamba/` | https://github.com/jbr-ai-labs/mamba.git | `2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f` | Upstream reference for the vendored in-process port `comparison/_mamba/` |
| `MAZero/` | https://github.com/liuqh16/MAZero.git | `b1c5084aafec9c9442e0ac8d6882935968be5818` | Pristine upstream of the method fork `hyper_mve/algo/mazero_mixed/` (GPL-3) |
| `DIMA/` | https://github.com/breez3young/DIMA | `3dcacaa80162cf6822bf5972b4e3ad4cb2e6ceb0` | Reference clone only (related-work; not wired into the registry) |

Permitted local edits (kept minimal, re-listed here when made):
- `HARL/`: relation env branch — `harl/utils/envs_tools.py` (+2 `elif`),
  `harl/envs/relation/{relation_env,relation_logger}.py` (new files),
  `harl/envs/__init__.py` (logger registry entry),
  `harl/configs/envs_cfgs/relation.yaml` (new file),
  `harl/utils/configs_tools.py` (`get_task_name` relation branch). *(phase 4 — DONE)*
- `MBOM/`: `utils/rl_utils.py` — `"MBAM"` type-name check → accepts `MBOM`;
  coin-game-specific `info` keys guarded. *(phase 5)*
- `m3w-marl/`, `mamba/`, `MAZero/`, `DIMA/`: **no edits.**
