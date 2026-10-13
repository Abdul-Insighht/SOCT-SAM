#!/bin/bash
# ==============================================================================
#                 Enhanced_RRSIS_UOT GPU Training Script (Optimized)
# ==============================================================================
# This script is highly optimized to run on high-performance GPU systems from scratch.
# It automatically applies all 4 core enhancements and utilizes naye SOTA hyperparameters.
#
# Target Performance: oIoU = 84.00%+, mIoU = 74.00%+
#
# Usage:
#   bash fine.sh [dataset_name] [data_root] [sam3_ckpt]
#
# Examples:
#   bash fine.sh rrsis_d /path/to/data ./pre-trained-weights/sam3.pt
# ==============================================================================

# Default parameters
DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
SAM3_CKPT=${3:-./pre-trained-weights/sam3.pt}
OUTPUT_DIR="./output/${DATASET}_enhanced_uot"

# Highlight setup information
echo "=============================================================================="
echo "🚀 Starting Enhanced_RRSIS_UOT training on GPU..."
echo "📊 Dataset:     ${DATASET}"
echo "📂 Data Root:   ${DATA_ROOT}"
echo "💾 SAM3 Ckpt:   ${SAM3_CKPT}"
echo "📁 Output Dir:  ${OUTPUT_DIR}"
echo "🔧 Optimizations: Dynamic LoRA | Contrastive Loss | Multi-Scale OT | OHEM Loss"
echo "=============================================================================="

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Ensure pre-trained weight exists or check parent dir fallback
if [ ! -f "${SAM3_CKPT}" ]; then
    echo "⚠️  [WARNING] Pre-trained weight not found at '${SAM3_CKPT}'"
    # Check parent dir fallback
    FALLBACK="../sam3_pre_trained_weights/sam3.pt"
    if [ -f "${FALLBACK}" ]; then
        echo "🔍 [INFO] Found fallback checkpoint in parent directory: '${FALLBACK}'"
        SAM3_CKPT=${FALLBACK}
    else
        echo "❌ [ERROR] Could not find pre-trained weights. Please make sure sam3.pt is placed correctly."
    fi
fi

# Run the training script with optimal hyperparameters for performance & speed from scratch
python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ${SAM3_CKPT} \
    --image_size 800 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 50 \
    --batch_size 4 \
    --grad_accum_steps 4\
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
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

echo "🎉 Training epoch run complete!"
