"""Run the full LoRA experiment sweep in one shot (sharing disabled).

Covers every required modelling situation x environment, scheduling them across a fixed pool
of GPUs (by default GPUs 2,3,4 -- GPUs 0 and 1 are intentionally never touched).

================================================================================
Experiment matrix  (share_subjective_trunk = False everywhere; output_rank LoRA r=32 always on)
================================================================================
3 modelling situations:
  * film_head, gen_scope OFF  -> preset *_film_lora       (film_head + output-LoRA)
  * film_head, gen_scope ON   -> preset *_film_lora_fc2   (film_head + output-LoRA + per-context fc2 delta)
  * base_gen                  -> preset *_base_lora       (base_gen  + output-LoRA)
2 environments (2 levels):
  * 2agent -> duo    presets (N=2, 1a+1b, random_walk)
  * 4agent -> medium presets (N=4, 2a+2b, static)
=> 3 situations x 2 environments = 6 runs.

================================================================================
Output directory layout  (--root, default runs/lora_sweep)
================================================================================
  <root>/
    2agent/
      film_head/
        off/   { tb/  ckpt/  train.log }     <- duo_film_lora
        on/    { tb/  ckpt/  train.log }     <- duo_film_lora_fc2
      basegen/ { tb/  ckpt/  train.log }     <- duo_base_lora
    4agent/
      film_head/
        off/   ...                           <- medium_film_lora
        on/    ...                           <- medium_film_lora_fc2
      basegen/ ...                           <- medium_base_lora

Per leaf: tb/  = TensorBoard scalars (train_main --log_dir),
          ckpt/ = checkpoints step_<N>.pt (train_main --ckpt_dir),
          train.log = full captured stdout/stderr (the printed step metrics).

================================================================================
GPU scheduling
================================================================================
Each training run is single-GPU (train_main uses cuda:0). To pin one physical GPU per run we
launch each subprocess with CUDA_VISIBLE_DEVICES=<one id> so it sees exactly one device as
cuda:0 -- this is more robust than exporting CUDA_VISIBLE_DEVICES=2,3,4 globally (which would
make every run land on the first visible GPU). The pool defaults to {2,3,4}, so up to 3 runs
execute concurrently; as each finishes its GPU is returned to the pool and the next run starts.

Examples
--------
    # full sweep, GPUs 2,3,4, full preset budget:
    python hyper_mve/scripts/run_lora_experiments.py

    # shorter smoke sweep (cap steps), custom output root:
    python hyper_mve/scripts/run_lora_experiments.py --max_steps 2000 --root runs/lora_smoke

    # only the 2-agent level, print the plan without launching:
    python hyper_mve/scripts/run_lora_experiments.py --envs 2agent --dry_run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Repo root = the OUTER hyper_mve/ dir (so `hyper_mve` imports work, mirroring train_main).
# __file__ = <root>/hyper_mve/scripts/run_lora_experiments.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_MAIN = os.path.join(REPO_ROOT, "hyper_mve", "scripts", "train_main.py")

# (env_level, situation_key) -> preset name. situation_key encodes the directory layout too.
PRESETS = {
    "2agent": {
        "film_off": "duo_film_lora",
        "film_on": "duo_film_lora_fc2",
        "basegen": "duo_base_lora",
        # [v4-opt 2026-06c] P1.2: B'(i) zero-shot protocol cell — same preset as
        # film_off but with c_mode forced to static via --override (so trained on a
        # fixed grid of c values, eval probes unseen c). Random-walk c only gives
        # 'mechanism survival' evidence for B'; static c is the proper extrapolation test.
        "film_off_static": "duo_film_lora",
    },
    "4agent": {
        "film_off": "medium_film_lora",
        "film_on": "medium_film_lora_fc2",
        "basegen": "medium_base_lora",
    },
}

# situation_key -> relative leaf path under <root>/<env>/. film_head splits on/off; base_gen flat.
LEAF_SUBPATH = {
    "film_off": os.path.join("film_head", "off"),
    "film_on": os.path.join("film_head", "on"),
    "basegen": "basegen",
    # [v4-opt 2026-06c] P1.2: keep the static-c cell visually distinct from the
    # main random-walk cells so any side-by-side analysis stays unambiguous.
    "film_off_static": os.path.join("film_head_static", "off"),
}

# [v4-opt 2026-06c] P1.2: per-cell train_main --override args, applied on top of
# the preset. Empty/missing key → no override.
SITUATION_OVERRIDES: dict[str, tuple[str, ...]] = {
    "film_off_static": ("env.c_mode=static",),
}

# The set of situations the sweep will fan out over by default (in launch order).
DEFAULT_SITUATIONS: tuple[str, ...] = ("basegen", "film_off", "film_on")

POLL_SECONDS = 5.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the full LoRA sweep (3 situations x 2 envs = 6 runs) across a GPU pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default="runs/lora_sweep",
                   help="output root; per-run results go under <root>/<env>/<model>[/<gen_scope>]")
    p.add_argument("--gpus", default="2,3,4",
                   help="comma-separated physical GPU ids to use as the pool (GPUs 0,1 are off-limits)")
    p.add_argument("--envs", default="2agent,4agent",
                   help="comma-separated subset of {2agent,4agent} to run")
    p.add_argument("--seed", type=int, default=None,
                   help="legacy single-seed flag (use --seeds for multi-seed). "
                        "If set, equivalent to --seeds <seed>.")
    # [v4-opt 2026-06c] P1.1: multi-seed fan-out. The eval-return std observed in
    # the 2agent dump was ~28 on a tail mean of ~52 -> single-seed verdicts are
    # within 1σ of each other across cells. ≥2-3 seeds per cell are needed for the
    # B'(iii) peak claim to be defensible (Review §3.5 hard-gate caveat).
    p.add_argument("--seeds", default=None,
                   help="comma-separated list of seeds, one run per (env × situation × seed). "
                        "Default: '0' (equivalent to previous --seed 0 behaviour).")
    # [v4-opt 2026-06c] P1.2: optionally add the static-c B'(i) protocol cell.
    p.add_argument("--include_static_b_prime", action="store_true",
                   help="add a film_head + c_mode=static cell (B'(i) zero-shot protocol) "
                        "for each enabled env (only 2agent supports it currently).")
    p.add_argument("--max_steps", type=int, default=None,
                   help="override train.max_train_steps for every run (default: use each preset's budget)")
    p.add_argument("--variant", default="hyper", help="train_main --variant")
    p.add_argument("--collect_planner", dest="collect_planner", action="store_true", default=True,
                   help="keep the MVE collection planner ON (default; required for the policy to learn)")
    p.add_argument("--no_collect_planner", dest="collect_planner", action="store_false",
                   help="disable the collection planner (debug only)")
    p.add_argument("--dry_run", action="store_true",
                   help="print the planned runs + commands and exit without launching anything")
    return p.parse_args(argv)


def parse_gpu_pool(gpus_str):
    """Parse '2,3,4' -> [2,3,4] with validation; GPUs 0 and 1 are rejected by policy."""
    pool = []
    for tok in gpus_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            gid = int(tok)
        except ValueError:
            raise SystemExit(f"[run_lora] invalid GPU id {tok!r} in --gpus {gpus_str!r}")
        if gid < 0:
            raise SystemExit(f"[run_lora] negative GPU id {gid} not allowed")
        if gid in (0, 1):
            raise SystemExit(
                f"[run_lora] GPU {gid} is off-limits (policy: never use GPU 0 or 1). "
                f"Use --gpus 2,3,4."
            )
        pool.append(gid)
    if not pool:
        raise SystemExit("[run_lora] empty GPU pool; pass e.g. --gpus 2,3,4")
    # de-dup while preserving order
    seen, uniq = set(), []
    for g in pool:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def _resolve_seeds(args) -> list[int]:
    """Resolve the seed list (P1.1).

    Precedence: ``--seeds`` (multi-seed) > ``--seed`` (legacy single) > default [0].
    Mixing them is an error so the user-facing semantics stay obvious.
    """
    if args.seeds is not None and args.seed is not None:
        raise SystemExit("[run_lora] pass either --seeds or --seed, not both")
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise SystemExit(f"[run_lora] empty --seeds list: {args.seeds!r}")
        return seeds
    if args.seed is not None:
        return [int(args.seed)]
    return [0]


def build_jobs(args):
    """Build the ordered list of run descriptors for the requested environments."""
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    for e in envs:
        if e not in PRESETS:
            raise SystemExit(f"[run_lora] unknown env {e!r} (valid: {sorted(PRESETS)})")

    seeds = _resolve_seeds(args)
    multi_seed = len(seeds) > 1

    situations = list(DEFAULT_SITUATIONS)
    if args.include_static_b_prime:
        situations.append("film_off_static")

    jobs = []
    # Order: env x situation x seed. Within each env, situations are ordered
    # base->film so the (usually heavier) base_gen runs start first; the optional
    # static-B' cell appends last. Seeds are the innermost loop so the GPU pool
    # round-robins through cells per seed rather than finishing seed=0 entirely.
    for env in envs:
        for situation in situations:
            if situation not in PRESETS[env]:
                # e.g. 4agent has no static-B' variant yet — silently skip.
                continue
            preset = PRESETS[env][situation]
            overrides = SITUATION_OVERRIDES.get(situation, ())
            for seed in seeds:
                seed_subdir = f"seed{seed}" if multi_seed else ""
                leaf = os.path.join(args.root, env, LEAF_SUBPATH[situation], seed_subdir)
                # When seed_subdir is empty, os.path.join leaves a trailing "" -> normpath cleans it.
                leaf = os.path.normpath(leaf)
                name = f"{env}/{situation}" + (f"/s{seed}" if multi_seed else "")
                jobs.append({
                    "name": name,
                    "env": env,
                    "situation": situation,
                    "preset": preset,
                    "seed": seed,
                    "overrides": overrides,
                    "leaf": leaf,
                    "log_dir": os.path.join(leaf, "tb"),
                    "ckpt_dir": os.path.join(leaf, "ckpt"),
                    "logfile": os.path.join(leaf, "train.log"),
                })
    return jobs


def build_command(job, args):
    """The train_main.py argv for a single run (GPU is pinned via env, not argv)."""
    cmd = [
        sys.executable, TRAIN_MAIN,
        "--preset", job["preset"],
        "--variant", args.variant,
        "--seed", str(job["seed"]),
        "--log_dir", job["log_dir"],
        "--ckpt_dir", job["ckpt_dir"],
    ]
    for ov in job.get("overrides", ()):
        cmd += ["--override", ov]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    if not args.collect_planner:
        cmd += ["--no_collect_planner"]
    return cmd


def launch(job, gpu, args):
    """Start one training subprocess pinned to a single physical GPU. Returns (proc, fh)."""
    os.makedirs(job["leaf"], exist_ok=True)
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["ckpt_dir"], exist_ok=True)

    env = os.environ.copy()
    # Pin exactly one physical GPU; the child then uses cuda:0 == this device.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Keep stdout unbuffered so train.log streams live.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_command(job, args)
    fh = open(job["logfile"], "w", encoding="utf-8")
    fh.write(f"# {job['name']}  preset={job['preset']}  GPU={gpu}\n")
    fh.write("# " + " ".join(cmd) + "\n\n")
    fh.flush()
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env,
        stdout=fh, stderr=subprocess.STDOUT,
    )
    print(f"[run_lora] START  {job['name']:18s} preset={job['preset']:22s} "
          f"GPU={gpu}  pid={proc.pid}  -> {job['leaf']}", flush=True)
    return proc, fh


def main(argv=None):
    args = parse_args(argv)
    gpu_pool = parse_gpu_pool(args.gpus)
    jobs = build_jobs(args)

    if not os.path.isfile(TRAIN_MAIN):
        raise SystemExit(f"[run_lora] cannot find train_main.py at {TRAIN_MAIN}")

    print(f"[run_lora] repo root : {REPO_ROOT}")
    print(f"[run_lora] output    : {os.path.abspath(args.root)}")
    print(f"[run_lora] GPU pool  : {gpu_pool}  (GPUs 0,1 excluded by policy)")
    print(f"[run_lora] runs ({len(jobs)}):")
    for j in jobs:
        print(f"    - {j['name']:18s} preset={j['preset']:22s} leaf={j['leaf']}")

    if args.dry_run:
        print("\n[run_lora] --dry_run: commands that WOULD run:")
        for j in jobs:
            print(f"  CUDA_VISIBLE_DEVICES=<gpu> " + " ".join(build_command(j, args)))
        return 0

    # ---- GPU-pool scheduler: keep <= len(gpu_pool) runs in flight ----
    free_gpus = list(gpu_pool)
    pending = list(jobs)
    running = {}   # gpu_id -> (job, proc, fh)
    results = {}   # job_name -> return code (or "launch-error")
    interrupted = False

    try:
        while pending or running:
            # Fill idle GPUs with pending jobs.
            while free_gpus and pending:
                gpu = free_gpus.pop(0)
                job = pending.pop(0)
                try:
                    proc, fh = launch(job, gpu, args)
                    running[gpu] = (job, proc, fh)
                except Exception as exc:  # noqa: BLE001 - never let one launch kill the sweep
                    print(f"[run_lora] LAUNCH-FAIL {job['name']}: {exc}", flush=True)
                    results[job["name"]] = "launch-error"
                    free_gpus.append(gpu)   # give the GPU back

            # Reap finished runs.
            for gpu, (job, proc, fh) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                fh.flush()
                fh.close()
                results[job["name"]] = rc
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[run_lora] DONE   {job['name']:18s} {status:12s} "
                      f"GPU={gpu} -> {job['logfile']}", flush=True)
                del running[gpu]
                free_gpus.append(gpu)

            if pending or running:
                time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_lora] interrupted -- terminating running runs...", flush=True)
        for gpu, (job, proc, fh) in running.items():
            proc.terminate()
            results.setdefault(job["name"], "interrupted")
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        # give them a moment, then hard-kill stragglers
        time.sleep(3.0)
        for gpu, (job, proc, fh) in running.items():
            if proc.poll() is None:
                proc.kill()

    # ---- Summary ----
    print("\n[run_lora] ===== summary =====")
    n_ok = sum(1 for v in results.values() if v == 0)
    for j in jobs:
        v = results.get(j["name"], "not-run")
        tag = "OK" if v == 0 else str(v)
        print(f"    {j['name']:18s} preset={j['preset']:22s} -> {tag}")
    print(f"[run_lora] {n_ok}/{len(jobs)} runs succeeded.")

    if interrupted:
        return 130
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
