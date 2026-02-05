#!/bin/bash
# Training job script for semantic segmentation with DINOv3 + UperNet
# Uses the preset-based train_semantic.py

# ============================================================================
# HUGGINGFACE LOGIN
# ============================================================================
# Set your HuggingFace token here or export it before running
# export HF_TOKEN='your_huggingface_token_here'

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set. Please set it to access DINOv3 models."
    echo "Run: export HF_TOKEN='your_token_here'"
    exit 1
fi

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data
DATA_ROOT="/workspace/data"
# Dataset
DATASET="cou" 
# Options: cou, uiis10k, usod10k, liaci, trashcan, etc.
#IMG_SIZE="536 960"        # Height Width -- Currently using default image size from DataModule

# Model - DINOv3 options:
#   upernet_dinov3_small  (21M params)
#   upernet_dinov3_base   (86M params)
#   upernet_dinov3_large  (300M params)
#   upernet_dinov3_7b     (6.7B params)
MODEL="upernet_dinov3_base"

# Training parameters
BATCH_SIZE=22              # <-- Adjust this while running
EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4


# Augmentation: none, light, medium, heavy, underwater
AUGMENTATION="medium"

# Training options
FREEZE_BACKBONE=true
SCHEDULER="cosine"
WARMUP_EPOCHS=0

# Hardware
GPUS="0"
PRECISION="16-mixed"
NUM_WORKERS=8
# Logging
WANDB_PROJECT="UnderwaterBenchmark"
LOG_DIR="/workspace/logs"

# ============================================================================
# BUILD COMMAND
# ============================================================================

CMD="python train_semantic.py"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --model $MODEL"
CMD="$CMD --data_root $DATA_ROOT"
#CMD="$CMD --img_size $IMG_SIZE"  # Currently using default image size from DataModule
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --lr $LEARNING_RATE"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --scheduler $SCHEDULER"
CMD="$CMD --warmup_epochs $WARMUP_EPOCHS"
CMD="$CMD --augmentation $AUGMENTATION"
CMD="$CMD --gpus $GPUS"
CMD="$CMD --precision $PRECISION"
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --log_dir $LOG_DIR"
CMD="$CMD --hf_login"

[ "$FREEZE_BACKBONE" = true ] && CMD="$CMD --freeze_backbone"
[ -n "$WANDB_PROJECT" ] && CMD="$CMD --wandb_project $WANDB_PROJECT"

# ============================================================================
# EXECUTE
# ============================================================================

echo "=============================================="
echo "Starting Training: $DATASET + $MODEL"
echo "=============================================="
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "=============================================="
echo ""

eval $CMD
