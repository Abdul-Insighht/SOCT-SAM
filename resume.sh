#!/bin/bash
# ==============================================================================
#                 Enhanced_RRSIS_UOT GPU Resume Training Script
# ==============================================================================
# This script is highly optimized to resume training from the best checkpoint
# with premium fine-tuning settings (Decay Floor and Regularization) to beat SOTA.
#
# Usage:
#   bash resume.sh [dataset_name] [data_root] [sam3_ckpt]
#
# Examples:
#   bash resume.sh rrsis_d ./data ./pre-trained-weights/sam3.pt
# ==============================================================================

# Default parameters (matching fine.sh exactly)
DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
SAM3_CKPT=${3:-./pre-trained-weights/sam3.pt}
OUTPUT_DIR="./output/${DATASET}_enhanced_uot"
RESUME_CKPT=${4:-"${OUTPUT_DIR}/latest_model.pth"}

# Auto-detect checkpoint if default does not exist
if [ ! -f "${RESUME_CKPT}" ] && [ ! -d "${RESUME_CKPT}" ]; then
    for cand in "${OUTPUT_DIR}/best_model.pth" "/kaggle/input/best_hr/best_model" "/kaggle/input/best-hr/best_model" "/kaggle/input/best_hr" "/kaggle/input/best-hr" "/kaggle/working/best_model.pth"; do
        if [ -f "${cand}" ] || [ -d "${cand}" ]; then
            RESUME_CKPT="${cand}"
            break
        fi
    done
fi

# Highlight setup information
echo "=============================================================================="
echo "Resuming Enhanced_RRSIS_UOT training on GPU..."
echo "Dataset:      ${DATASET}"
echo "Data Root:    ${DATA_ROOT}"
echo "SAM3 Ckpt:    ${SAM3_CKPT}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Resume Ckpt:  ${RESUME_CKPT}"
echo "Optimizations: Cosine LR Floor | Weight Decay 0.03 | Fine-tuning LR"
echo "=============================================================================="

# Ensure the resume checkpoint exists
if [ ! -f "${RESUME_CKPT}" ] && [ ! -d "${RESUME_CKPT}" ]; then
    echo "[ERROR] Could not find the resume checkpoint file or directory at '${RESUME_CKPT}'!"
    exit 1
fi

# Run the training script with optimal hyperparameters for performance & speed
python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ${SAM3_CKPT} \
    --resume ${RESUME_CKPT} \
    --image_size 800 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 50 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 2e-5 \
    --lr_backbone 5e-6 \
    --lr_decoder 2e-5 \
    --weight_decay 0.03 \
    --warmup_epochs 5 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    --contrastive_weight 0.1 \
    --ohem_hard_ratio 0.3 \
    --ot_reg 0.1 \
    --ot_num_iter 10 \
    --num_ot_scales 3 \
    --use_soct \
    --lambda_ord 5.0 \
    --lambda_occ 1.0 \
    --lambda_rho 0.5 \
    --focal_gamma 2.0

echo "Resumed training epoch run complete!"
