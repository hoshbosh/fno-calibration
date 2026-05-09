#!/usr/bin/env bash

# Smoke test - go through the entire pipeline to verify it all works before we do higher scale stuff

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="configs/smoke.yaml"
EPOCHS=20

TRAIN="outputs/smoke_train.h5"
VAL="outputs/smoke_val.h5"
TEST="outputs/smoke_test.h5"
OOD="outputs/smoke_ood.h5"

echo "=== [1/4] Generating smoke datasets (1k/200/200/100 surfaces) ==="
# Skip regeneration if files exist — generator is the slow step.
gen() {
    local out="$1"; local n="$2"; local seed="$3"
    if [[ -f "$out" ]]; then
        echo "    $out exists — skipping. Delete to force regeneration."
    else
        python -m data.generate_heston --config "$CONFIG" \
            --n_samples "$n" --seed "$seed" --output "$out"
    fi
}
gen "$TRAIN" 1000 42
gen "$VAL"    200 43
gen "$TEST"   200 44
gen "$OOD"    100 45

echo "=== [2/4] Training FNO ($EPOCHS epochs) ==="
python train.py --config "$CONFIG" --model fno --epochs "$EPOCHS" --wandb_mode online

echo "=== [3/4] Training MLP ($EPOCHS epochs) ==="
python train.py --config "$CONFIG" --model mlp --epochs "$EPOCHS" --wandb_mode online

echo "=== [4/4] Evaluating both models ==="
python eval.py --config "$CONFIG" --checkpoint checkpoints/fno_best.pt
python eval.py --config "$CONFIG" --checkpoint checkpoints/mlp_best.pt

echo
echo "=== Smoke test complete ==="
