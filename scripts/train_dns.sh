#!/bin/bash
# FlowRefiner training on TGV (Taylor-Green vortex, 64x128x128).
#
# Two training modes:
#   MODE=two_stage  K=0 pretrain (150 ep) -> K=2 fixed_range fine-tune (50 ep)
#   MODE=joint      K=2 fixed_range from scratch (200 ep)
#
# Usage:
#   DATA_DIR=/path/to/TGV_data MODE=two_stage bash scripts/train_dns.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/TGV_data}"
OUT_ROOT="${OUT_ROOT:-checkpoints}"
MODE="${MODE:-two_stage}"

K=2
SIGMA_SCHED=fixed_range
SIGMA_MAX=0.01
SIGMA_MIN=0.001
ODE_STEPS=2

COMMON=(
    --data_source dns --data_dir "$DATA_DIR"
    --refiner_steps "$K"
    --sigma_schedule "$SIGMA_SCHED"
    --sigma_max "$SIGMA_MAX" --sigma_min "$SIGMA_MIN"
    --ode_steps "$ODE_STEPS"
    --batch_size 1 --lr 1e-4
    --use_wandb --wandb_project flowrefiner
)

case "$MODE" in
  two_stage)
    K0_DIR="${OUT_ROOT}/flowrefiner_dns_K0_pretrain"
    python train.py "${COMMON[@]}" \
        --refiner_steps 0 --sigma_schedule ddpm \
        --epochs 150 --checkpoint_dir "$K0_DIR"

    FT_DIR="${OUT_ROOT}/flowrefiner_dns_K${K}_${SIGMA_SCHED}_ft"
    python train.py "${COMMON[@]}" \
        --epochs 50 \
        --checkpoint_dir "$FT_DIR" \
        --finetune "${K0_DIR}/latest.pt"
    echo "[two_stage] Best checkpoint: ${FT_DIR}/best.pt"
    ;;

  joint)
    JOINT_DIR="${OUT_ROOT}/flowrefiner_dns_K${K}_${SIGMA_SCHED}_joint"
    python train.py "${COMMON[@]}" \
        --epochs 200 --checkpoint_dir "$JOINT_DIR"
    echo "[joint] Best checkpoint: ${JOINT_DIR}/best.pt"
    ;;

  *)
    echo "Unknown MODE: ${MODE}. Use 'two_stage' or 'joint'." >&2
    exit 1
    ;;
esac
