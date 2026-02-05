# Training Job Examples

This document provides example configurations for different training scenarios using `trainjobs.sh`.

## Quick Start

Make the script executable:
```bash
chmod +x trainjobs.sh
```

Run with default settings:
```bash
./trainjobs.sh
```

## Example Configurations

### 1. Quick Test Run (Fast Development)

Edit `trainjobs.sh` and set:
```bash
FAST_DEV_RUN=true
EPOCHS=1
BATCH_SIZE=2
CACHE_DATASET=false
AUGMENTATION="none"
```

### 2. Production Training with WandB

```bash
# In trainjobs.sh, set:
AUGMENTATION="underwater"
EPOCHS=100
BATCH_SIZE=30
PRECISION="16-mixed"

WANDB_PROJECT="underwater-segmentation"
WANDB_ENTITY="your-team"
EXPERIMENT_NAME="vit-base-upernet-run1"

# Before running:
export WANDB_API_KEY='your_wandb_api_key'
./trainjobs.sh
```

### 3. Multi-GPU Training (DDP)

```bash
# In trainjobs.sh, set:
GPUS="0,1,2,3"  # Use 4 GPUs
STRATEGY="ddp"
BATCH_SIZE=8  # Per GPU (effective batch = 8 * 4 = 32)
NUM_WORKERS=4  # Per GPU

./trainjobs.sh
```

### 4. Memory-Efficient Training (Small GPU)

```bash
# In trainjobs.sh, set:
BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=8  # Effective batch = 4 * 8 = 32
PRECISION="16-mixed"
CACHE_DATASET=false
NUM_WORKERS=2

./trainjobs.sh
```

### 5. DINOv2 Backbone

```bash
# In trainjobs.sh, set:
BACKBONE="facebook/dinov2-base"
BACKBONE_INDICES="2 5 8 11"  # DINOv2-Base has 12 layers
SCALES="4 2 1 0.5"
NUM_REGISTER_TOKENS=4  # DINOv2 uses 4 register tokens
INTERPOLATE_POS_ENCODING=true  # Will use native interpolation

./trainjobs.sh
```


#### 5.1 DINOV3 EXample
```bash
# Model configuration
BACKBONE="facebook/dinov3-vitb16-pretrain-lvd1689m"  # Options: google/vit-*, facebook/dinov2-*, facebook/dino-*
BACKBONE_INDICES="2 5 8 11"  # For ViT-Base (12 layers). Use "5 11 17 23" for ViT-Large (24 layers)
SCALES="4 2 1 0.5"
NUM_CLASSES=""  # Auto-detect if not specified
UPERNET_HIDDEN_SIZE=512
OUT_CHANNELS=768
NUM_REGISTER_TOKENS=4  # Set to 4 for DINOv2 with registers
INTERPOLATE_POS_ENCODING=true  # Enable positional encoding interpolation for arbitrary image sizes


./trainjobs.sh
```


### 6. Large Model (ViT-Large)

```bash
# In trainjobs.sh, set:
BACKBONE="google/vit-large-patch16-224"
BACKBONE_INDICES="5 11 17 23"  # ViT-Large has 24 layers
BATCH_SIZE=8  # Reduce for large model
OUT_CHANNELS=1024  # Larger for ViT-Large
UPERNET_HIDDEN_SIZE=1024

./trainjobs.sh
```

### 7. Transfer Learning (Frozen Backbone)

```bash
# In trainjobs.sh, set:
FREEZE_BACKBONE=true
LEARNING_RATE=1e-3  # Higher LR for head-only training
EPOCHS=50
WARMUP_EPOCHS=2

./trainjobs.sh
```

### 8. Resume Training from Checkpoint

```bash
# In trainjobs.sh, set:
RESUME_FROM="./logs/checkpoints/last.ckpt"
EPOCHS=200  # Continue for more epochs

./trainjobs.sh
```

### 9. Debugging - Overfit on Small Batch

```bash
# In trainjobs.sh, set:
OVERFIT_BATCHES=10  # Use only 10 batches
EPOCHS=1000
AUGMENTATION="none"
LOG_EVERY_N_STEPS=1

./trainjobs.sh
```

### 10. Heavy Augmentation for Robustness

```bash
# In trainjobs.sh, set:
AUGMENTATION="heavy"
EPOCHS=150
SCHEDULER="cosine"
WARMUP_EPOCHS=10

./trainjobs.sh
```

## Command-Line Overrides

You can override any variable from the command line:

```bash
# Quick single-variable override
BATCH_SIZE=16 ./trainjobs.sh

# Multiple overrides
BATCH_SIZE=16 EPOCHS=50 AUGMENTATION="heavy" ./trainjobs.sh

# WandB configuration
WANDB_PROJECT="my-project" WANDB_API_KEY="mykey" ./trainjobs.sh
```

## Docker Usage

### Run in Docker Container

```bash
docker run --gpus all \
    --shm-size=50gb \
    -v $(pwd):/workspace/code \
    -v /path/to/data:/workspace/data \
    -e WANDB_API_KEY='your_key' \
    underwater-benchmark:latest \
    bash /workspace/code/trainjobs.sh
```

### Interactive Docker Session

```bash
docker run -it --gpus all \
    --shm-size=50gb \
    -v $(pwd):/workspace/code \
    -v /path/to/data:/workspace/data \
    underwater-benchmark:latest \
    /bin/bash

# Inside container:
cd /workspace/code
chmod +x trainjobs.sh
./trainjobs.sh
```

## Performance Tuning Tips

### For Maximum Speed

```bash
CACHE_DATASET=true  # If you have 50GB+ RAM
NUM_WORKERS=8  # 4-8 per GPU is optimal
PERSISTENT_WORKERS=true
PIN_MEMORY=true
PREFETCH_FACTOR=4
PRECISION="16-mixed"
```

### For Maximum Accuracy

```bash
AUGMENTATION="underwater"  # Task-specific augmentation
BATCH_SIZE=64  # Larger if possible (use gradient accumulation)
SCHEDULER="cosine"
WARMUP_EPOCHS=10
EARLY_STOPPING=15
```

### For Limited Resources

```bash
BATCH_SIZE=2
ACCUMULATE_GRAD_BATCHES=16  # Effective batch = 32
CACHE_DATASET=false
NUM_WORKERS=2
PRECISION="16-mixed"
```

## Common Issues

### 1. Out of Memory (GPU)
- Reduce `BATCH_SIZE`
- Set `PRECISION="16-mixed"`
- Increase `ACCUMULATE_GRAD_BATCHES`
- Reduce `IMG_SIZE`

### 2. Out of Memory (RAM)
- Set `CACHE_DATASET=false`
- Reduce `NUM_WORKERS`
- In Docker, increase `--shm-size`

### 3. Slow Training
- Increase `NUM_WORKERS`
- Set `CACHE_DATASET=true` if RAM available
- Use `PRECISION="16-mixed"`
- Check `PERSISTENT_WORKERS=true`

### 4. Poor Convergence
- Try `AUGMENTATION="medium"` or `"underwater"`
- Increase `WARMUP_EPOCHS`
- Adjust `LEARNING_RATE` (try 5e-5 to 2e-4)
- Check `INCREASE_IDX` and `FILL_BACKGROUND` match your dataset

## Model-Specific Settings

### Google ViT
```bash
BACKBONE="google/vit-base-patch16-224"
BACKBONE_INDICES="2 5 8 11"
NUM_REGISTER_TOKENS=0
INTERPOLATE_POS_ENCODING=true  # Uses manual interpolation
```

### Facebook DINOv2
```bash
BACKBONE="facebook/dinov2-base"
BACKBONE_INDICES="2 5 8 11"
NUM_REGISTER_TOKENS=4
INTERPOLATE_POS_ENCODING=true  # Uses native interpolation
```

### Facebook DINO
```bash
BACKBONE="facebook/dino-vitb16"
BACKBONE_INDICES="2 5 8 11"
NUM_REGISTER_TOKENS=0
INTERPOLATE_POS_ENCODING=true  # Uses native interpolation
```

## Monitoring Training

### Check Logs
```bash
tail -f logs/training.log
```

### Monitor with WandB
Visit: https://wandb.ai/your-entity/your-project

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Checkpoints
```bash
ls -lh logs/checkpoints/
```



