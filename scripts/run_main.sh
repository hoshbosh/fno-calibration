#!/usr/bin/env bash

# Phase 2 main runs: FNO + MLP, 3 seeds each, 200 epochs on full 100k data.
# Sequential (single GPU). For parallel multi-GPU, split across workers manually.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="configs/default.yaml"
SEEDS=(42 43 44)
WANDB_MODE="${WANDB_MODE:-online}"   # override with `WANDB_MODE=offline ./scripts/run_main.sh`

run() {
    local model="$1"; local seed="$2"
    local ckpt="checkpoints/${model}_seed${seed}_best.pt"
    if [[ -f "$ckpt" ]]; then
        echo "    $ckpt exists — skipping. Delete to force retraining."
        return
    fi
    echo "=== ${model} | seed ${seed} ==="
    python train.py --config "$CONFIG" --model "$model" --seed "$seed" --wandb_mode "$WANDB_MODE"
}

echo "=== Phase 2 main runs (3 seeds × 2 models) ==="
for seed in "${SEEDS[@]}"; do
    run fno "$seed"
done
for seed in "${SEEDS[@]}"; do
    run mlp "$seed"
done

echo
echo "=== All main runs complete ==="
echo "Checkpoints:"
ls -1 checkpoints/{fno,mlp}_seed*_best.pt 2>/dev/null || true
