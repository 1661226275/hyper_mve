#!/bin/bash
# Recurring accurate 30-ep eval of MAMBA's accruing 2M-run checkpoints.
# Resumable: eval_mamba_checkpoints.py skips points already in the JSON.
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate lightzero
cd /home/data/zhengwenbo/hyper_mve
for i in $(seq 1 60); do
  CUDA_VISIBLE_DEVICES=2 python hyper_mve/scripts/eval_mamba_checkpoints.py \
    --run-dir runs/suite/rel_gate_duo/external_mamba_seed0_row0_be184580 \
    --preset rel_duo --episodes 30 \
    --out runs/_analysis/rel_defense/mamba_ckpt_series_gate.json 2>&1 | grep "^\[eval_mamba"
  CUDA_VISIBLE_DEVICES=2 python hyper_mve/scripts/eval_mamba_checkpoints.py \
    --run-dir runs/suite/rel_zero_shot_duo/external_mamba_seed0_row0_31eb304a \
    --preset rel_duo_holdout --episodes 30 \
    --out runs/_analysis/rel_defense/mamba_ckpt_series_zeroshot.json 2>&1 | grep "^\[eval_mamba"
  sleep 1800
done
