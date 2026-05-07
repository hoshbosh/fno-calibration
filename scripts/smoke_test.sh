#!/usr/bin/env bash

# Smoke test - go through the entire pipeline to verify it all works before we do higher scale stuff

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="configs/default.yaml"
DATA_PATH="outputs/smoke-test.h5"
EPOCHS=20

echo "=== [1/4] Generating dataset (1k surfaces) ==="                                                    
# Skip regeneration if the HDF5 already exists — the generator is the slow step                        
# (~5–10 min) and the dataset is deterministic given the seed.                                           
if [[ -f "$DATA_PATH" ]]; then                                                                           
    echo "    $DATA_PATH already exists — skipping. Delete to force regeneration."                       
else                                                                                                     
    python -m data.generate_heston --config "$CONFIG"                                                    
fi                                                                                                     
                                                                                                         
echo "=== [2/4] Training FNO ($EPOCHS epochs) ==="                                                       
python train.py --config "$CONFIG" --model fno --epochs "$EPOCHS" --wandb_mode disabled
                                                                                                         
echo "=== [3/4] Training MLP ($EPOCHS epochs) ==="                                                     
python train.py --config "$CONFIG" --model mlp --epochs "$EPOCHS" --wandb_mode disabled                  
                                                                                                         
echo "=== [4/4] Evaluating both models ==="
python eval.py --config "$CONFIG" --checkpoint checkpoints/fno_best.pt                                   
python eval.py --config "$CONFIG" --checkpoint checkpoints/mlp_best.pt                                   
                                                                                                         
echo                                                                                                     
echo "=== Smoke test complete ==="
