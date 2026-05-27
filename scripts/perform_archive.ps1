# perform_archive.ps1 — v4.7 → hyper_mve/_legacy_v4_7/ archive helper.
#
# This script is idempotent: re-running it after a partial archive prints
# "SKIP (missing)" for files that have already moved. It is meant for re-
# running the Pkg-01 archive step on a fresh clone — the main archive is
# already in git history and does not need to be re-run.
#
# Run from the repository root (the dir that contains hyper_mve/):
#     pwsh scripts/perform_archive.ps1

$ErrorActionPreference = "Stop"

$ARCHIVED = @(
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
    "config.py"
)

foreach ($f in $ARCHIVED) {
    $src = "hyper_mve/$f"
    $dst = "hyper_mve/_legacy_v4_7/$f"

    if (-not (Test-Path $src)) {
        Write-Host "SKIP (missing): $src"
        continue
    }

    $dstDir = Split-Path $dst -Parent
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Force $dstDir | Out-Null
    }

    git mv $src $dst
    Write-Host "Archived: $src -> $dst"
}

# Ensure every subdir under _legacy_v4_7/ has __init__.py so imports work.
Get-ChildItem hyper_mve/_legacy_v4_7 -Recurse -Directory | ForEach-Object {
    $init = Join-Path $_.FullName "__init__.py"
    if (-not (Test-Path $init)) {
        New-Item -ItemType File $init | Out-Null
    }
}

Write-Host ""
Write-Host "Archive complete. Run: python scripts/audit_legacy.py"
