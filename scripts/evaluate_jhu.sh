#!/bin/bash
# Run 3-round AR evaluation on the FIT (JHU) test split.
# Rebuild the scheduler with the SAME flags used at training time.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/JHU_DNS128}"
CKPT_DIR="${CKPT_DIR:-checkpoints/flowrefiner_jhu_K2_fixed_range_ft}"
OUT_DIR="${OUT_DIR:-results}"

for CKPT in best latest; do
    python evaluate.py \
        --data_source jhu --data_dir "$DATA_DIR" \
        --checkpoint_dir "$CKPT_DIR" --ckpt_type "$CKPT" \
        --refiner_steps 2 --sigma_schedule fixed_range \
        --sigma_max 0.01 --sigma_min 0.001 --ode_steps 2 \
        --max_ar_rounds 3 \
        --out_dir "$OUT_DIR"
done
