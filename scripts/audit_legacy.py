"""Audit Pkg-01 v4.7 → ``_legacy_v4_7/`` archive completeness.

Exit code 0 on success; non-zero with a diagnosis on any missing or leaked
file. Invoked as a Pkg-01 acceptance criterion (see SDD §6.1).

Usage (from the repository root)::

    python scripts/audit_legacy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Files that should now live under hyper_mve/_legacy_v4_7/ (relative paths).
ARCHIVED_FILES: tuple[str, ...] = (
    "envs/__init__.py",
    "envs/non_stationary_tag.py",
    "envs/ns_environment.py",
    "envs/make_env.py",
    "multiagent/__init__.py",
    "multiagent/core.py",
    "multiagent/environment.py",
    "multiagent/multi_discrete.py",
    "multiagent/rendering.py",
    "multiagent/scenario.py",
    "models/baseline_model.py",
    "models/hyper_muzero_model.py",
    "models/infer_muzero_model.py",
    "models/context_encoder.py",
    "models/gru_context_encoder.py",
    "models_advanced/__init__.py",
    "models_advanced/oracle_v2.py",
    "models_advanced/infer_v2.py",
    "models_advanced/chunked_hyper_network.py",
    "scripts/__init__.py",
    "scripts/train_baseline.py",
    "scripts/train_oracle.py",
    "scripts/train_oracle_v2.py",
    "scripts/train_infer.py",
    "scripts/train_infer_v2.py",
    "scripts/test_env.py",
    "scripts/test_discrete_env.py",
    "training/buffer.py",
    "training/mve.py",
    "config.py",
)

# Files that must remain in the main hyper_mve/ path (Pkg-04/05 will rewrite).
MAIN_PATH_PRESERVED: tuple[str, ...] = (
    "models/functional_nets.py",
    "models/hyper_network.py",
    "models/representation_net.py",
    "models/interfaces.py",
    "planning/__init__.py",
    "planning/mve_planner.py",
    "training/__init__.py",
    "training/muzero_trainer.py",
    "training/episode_buffer.py",
    "training/worker.py",
    "utils/__init__.py",
    "utils/utils.py",
    "utils/evaluator.py",
)

# Required helper files in the archive root.
AUX_REQUIRED: tuple[str, ...] = ("__init__.py", "README.md")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent / "hyper_mve"
    archive_root = project_root / "_legacy_v4_7"

    missing_archived = [f for f in ARCHIVED_FILES
                        if not (archive_root / f).exists()]
    missing_main = [f for f in MAIN_PATH_PRESERVED
                    if not (project_root / f).exists()]
    missing_aux = [f for f in AUX_REQUIRED
                   if not (archive_root / f).exists()]
    # Files that should have been moved but are still in the main path.
    leaked_in_main = [f for f in ARCHIVED_FILES
                      if (project_root / f).exists()]

    if missing_archived or missing_main or missing_aux or leaked_in_main:
        print("ARCHIVE AUDIT FAILED:")
        if missing_archived:
            print(f"  Missing from archive: {missing_archived}")
        if missing_main:
            print(f"  Missing from main path: {missing_main}")
        if missing_aux:
            print(f"  Missing aux files in archive: {missing_aux}")
        if leaked_in_main:
            print(f"  Leaked (still in main, should be archived): {leaked_in_main}")
        return 1

    print(
        f"ALL FILES ACCOUNTED FOR: {len(ARCHIVED_FILES)} archived, "
        f"{len(MAIN_PATH_PRESERVED)} in main path"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
