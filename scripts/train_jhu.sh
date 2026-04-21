#!/bin/bash
# FlowRefiner training on FIT (JHU forced isotropic turbulence, 128^3).
#
# Two training modes:
#   MODE=two_stage  (recommended, matches the paper)
#                   K=0 pretrain (150 ep) -> K=2 fixed_range fine-tune (50 ep)
#   MODE=joint      K=2 fixed_range from scratch (200 ep)
#
# Usage:
#   DATA_DIR=/path/to/JHU_DNS128 MODE=two_stage bash scripts/train_jhu.sh
#   DATA_DIR=/path/to/JHU_DNS128 MODE=joint     bash scripts/train_jhu.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/JHU_DNS128}"
OUT_ROOT="${OUT_ROOT:-checkpoints}"
MODE="${MODE:-two_stage}"

# ------------------------------- main hyper-parameters -------------------------------
K=2
SIGMA_SCHED=fixed_range
SIGMA_MAX=0.01
SIGMA_MIN=0.001
ODE_STEPS=2          # paper main setting; N=2 ODE substeps suffice

COMMON=(
    --data_source jhu --data_dir "$DATA_DIR"
    --refiner_steps "$K"
    --sigma_schedule "$SIGMA_SCHED"
    --sigma_max "$SIGMA_MAX" --sigma_min "$SIGMA_MIN"
    --ode_steps "$ODE_STEPS"
    --batch_size 1 --lr 1e-4
    --use_wandb --wandb_project flowrefiner
)

case "$MODE" in
  two_stage)
    # ---- Stage 1: K=0 pre-training (150 epochs) ----
    K0_DIR="${OUT_ROOT}/flowrefiner_jhu_K0_pretrain"
    python train.py "${COMMON[@]}" \
        --refiner_steps 0 --sigma_schedule ddpm \
        --epochs 150 --checkpoint_dir "$K0_DIR"

    # ---- Stage 2: K=2 fixed_range fine-tune (50 epochs) ----
    FT_DIR="${OUT_ROOT}/flowrefiner_jhu_K${K}_${SIGMA_SCHED}_ft"
    python train.py "${COMMON[@]}" \
        --epochs 50 \
        --checkpoint_dir "$FT_DIR" \
        --finetune "${K0_DIR}/latest.pt"

    echo "[two_stage] Best checkpoint: ${FT_DIR}/best.pt"
    ;;

  joint)
    # ---- K=2 fixed_range from scratch (200 epochs) ----
    JOINT_DIR="${OUT_ROOT}/flowrefiner_jhu_K${K}_${SIGMA_SCHED}_joint"
    python train.py "${COMMON[@]}" \
        --epochs 200 --checkpoint_dir "$JOINT_DIR"

    echo "[joint] Best checkpoint: ${JOINT_DIR}/best.pt"
    ;;

  *)
    echo "Unknown MODE: ${MODE}. Use 'two_stage' or 'joint'." >&2
    exit 1
    ;;
esac
