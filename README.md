# Underwater Benchmark - Semantic Segmentation

A PyTorch Lightning training framework for semantic segmentation using UperNet with Vision Transformer (ViT) backbones, optimized for underwater imagery.

## Features

- **ViT Backbone Support**: Any HuggingFace ViT model (google/vit-*, facebook/dino-*, facebook/dinov2-*)
- **GPU-Accelerated Augmentations**: Kornia-based augmentations running on GPU for maximum speed
- **Multi-scale Feature Extraction**: Extract features from multiple ViT layers at different spatial scales
- **FFCV Support**: Ultra-fast data loading (Docker setup includes FFCV pre-installed)
- **Advanced Training**: Mixed precision, multi-GPU, gradient accumulation
- **Rich Logging**: WandB integration with prediction visualizations
- **Dataset Caching**: Shared memory caching for fast training
- **Flexible Deployment**: Run with Docker or native Python environment

## Table of Contents

- [Quick Start (Docker)](#quick-start-docker)
- [Native Installation](#native-installation)
- [Training Examples](#training-examples)
- [Configuration Options](#configuration-options)
- [Dataset Format](#dataset-format)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Docker)

### Prerequisites
- Docker with NVIDIA GPU support (`nvidia-docker2`)
- NVIDIA drivers installed on host

### Build the Image

```bash
cd /home/tglobisch/code/underwaterbenchmark
docker build -t underwater-benchmark:latest -f docker/Dockerfile .
```

### Run Training in Docker

```bash
docker run --gpus all \
    --shm-size=50gb \
    -v $(pwd):/workspace/code \
    -v /path/to/data:/workspace/data \
    -v /path/to/logs:/workspace/logs \
    underwater-benchmark:latest \
    python /workspace/code/train_sementic.py \
        --data_root /workspace/data/coco \
        --image_folder images \
        --train_ann train_annotations.json \
        --val_ann val_annotations.json \
        --backbone google/vit-base-patch16-224 \
        --img_size 512 512 \
        --batch_size 8 \
        --epochs 100
```

### Interactive Development (Jupyter)

```bash
docker run --gpus all \
    --shm-size=50gb \
    -v $(pwd):/workspace/code \
    -p 8888:8888 \
    underwater-benchmark:latest
```

Access Jupyter Lab at: `http://localhost:8888`

**See [docker/README.md](docker/README.md) for detailed Docker usage.**

---

## Native Installation

### 1. System Requirements

- Python 3.10+
- CUDA 11.8 or later
- 16GB+ RAM (50GB+ recommended for dataset caching)

### 2. Install PyTorch

```bash
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio
```

### 3. Install Project Dependencies

```bash
cd /home/tglobisch/code/underwaterbenchmark
pip install -r requirements.txt
```

### 4. (Optional) Install FFCV for Fast Data Loading

```bash
# Install FFCV dependencies
pip install cupy-cuda11x numba

# Install FFCV
git clone https://github.com/libffcv/ffcv.git
cd ffcv
pip install .
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
python -c "import kornia; print(f'Kornia: {kornia.__version__}')"
```

---

## Training Examples

### Basic Training

```bash
python train_sementic.py \
    --data_root /path/to/coco \
    --image_folder images \
    --train_ann train_annotations.json \
    --val_ann val_annotations.json \
    --backbone google/vit-base-patch16-224 \
    --img_size 512 512 \
    --batch_size 8 \
    --epochs 100 \
    --increase_idx \
    --fill_background
```

### With GPU Augmentations

```bash
python train_sementic.py \
    --data_root /path/to/coco \
    --augmentation medium \
    --backbone google/vit-base-patch16-224 \
    --img_size 512 512 \
    --batch_size 8 \
    --epochs 100
```

Augmentation presets:
- `none` - No augmentation
- `light` - Minimal augmentation (horizontal flip, light color jitter)
- `medium` - Balanced augmentation (default, recommended)
- `heavy` - Aggressive augmentation (rotation, scaling, strong color jitter)
- `underwater` - Specialized for underwater imagery

### Multi-GPU Training

```bash
# Automatic multi-GPU with DDP
python train_sementic.py \
    --data_root /path/to/coco \
    --gpus auto \
    --strategy ddp \
    --batch_size 16 \
    --epochs 100

# Specific GPUs
python train_sementic.py \
    --data_root /path/to/coco \
    --gpus 0,1,2,3 \
    --strategy ddp \
    --batch_size 16 \
    --epochs 100
```

### With WandB Logging

First, login to WandB:

```bash
# Option 1: Interactive login (will open browser)
wandb login

# Option 2: Using API key directly
export WANDB_API_KEY='your_wandb_api_key_here'
wandb login $WANDB_API_KEY

# Option 3: Set environment variable only (no wandb login needed)
export WANDB_API_KEY='your_wandb_api_key_here'
```

Then run training:

```bash
python train_sementic.py \
    --data_root /path/to/coco \
    --wandb_project underwater-segmentation \
    --wandb_entity your-team \
    --experiment_name vit-upernet-run1 \
    --epochs 100
```

**For Docker:** Pass the WandB API key as an environment variable:

```bash
docker run --gpus all \
    --shm-size=50gb \
    -e WANDB_API_KEY='your_wandb_api_key_here' \
    -v $(pwd):/workspace/code \
    -v /path/to/data:/workspace/data \
    underwater-benchmark:latest \
    python /workspace/code/train_sementic.py \
        --data_root /workspace/data/coco \
        --wandb_project underwater-segmentation \
        --epochs 100
```

---

## Configuration Options

### Model Architecture

| Argument | Description | Default |
|----------|-------------|---------|
| `--backbone` | HuggingFace ViT model name | `google/vit-base-patch16-224` |
| `--backbone_indices` | Layer indices to extract features | `[2, 5, 8, 11]` |
| `--scales` | Spatial scales for each level | `[4.0, 2.0, 1.0, 0.5]` |
| `--out_channels` | Output channels per level | `512` |
| `--upernet_hidden_size` | Hidden size for UperNet decoder | `512` |
| `--num_register_tokens` | Number of register tokens (DINOv2) | `0` |
| `--use_auxiliary_head` | Use auxiliary segmentation head | `False` |
| `--freeze_backbone` | Freeze backbone weights | `False` |

### Data Augmentation

| Argument | Description | Default |
|----------|-------------|---------|
| `--augmentation` | Augmentation preset (none/light/medium/heavy/underwater) | `medium` |
| `--aug_prob` | Base probability for augmentations | `0.5` |
| `--aug_rotation` | Random rotation degrees | `10.0` |
| `--aug_horizontal_flip` | Enable horizontal flip | `True` |
| `--aug_vertical_flip` | Enable vertical flip | `False` |
| `--aug_color_jitter` | Enable color jittering | `True` |

### Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | `100` |
| `--batch_size` | Batch size per GPU | `8` |
| `--lr` | Learning rate | `1e-4` |
| `--weight_decay` | Weight decay | `1e-4` |
| `--scheduler` | LR scheduler (cosine/step/polynomial/none) | `cosine` |
| `--warmup_epochs` | Number of warmup epochs | `5` |
| `--early_stopping` | Early stopping patience | `None` |

### Hardware

| Argument | Description | Default |
|----------|-------------|---------|
| `--gpus` | GPUs to use (e.g., '0', '0,1', 'auto') | `auto` |
| `--precision` | Training precision (16-mixed/bf16-mixed/32) | `32` |
| `--strategy` | Distributed strategy (ddp/auto) | `auto` |
| `--accumulate_grad_batches` | Gradient accumulation steps | `1` |

### Data Loading

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_workers` | Number of data loading workers | `4` |
| `--cache_dataset` | Cache dataset in shared memory | `False` |
| `--pin_memory` | Pin memory for faster GPU transfer | `True` |
| `--persistent_workers` | Keep workers alive between epochs | `True` |

---

## Dataset Format

Your dataset should follow COCO format:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── train_annotations.json
    ├── val_annotations.json
    └── test_annotations.json  # Optional
```

### Label Configuration

Use these flags to configure label handling:

- `--increase_idx`: Increase class indices by 1 (if your labels start from 0 but you want background=0, class1=1, etc.)
- `--fill_background`: Fill unlabeled pixels as background class
- `--ignore_index`: Label index to ignore in loss computation (default: 255)

---

## Advanced Usage

### ViT Backbone Options

#### Supported Models

```bash
# Google ViT
--backbone google/vit-base-patch16-224
--backbone google/vit-large-patch16-224

# Facebook DINO
--backbone facebook/dino-vitb16
--backbone facebook/dino-vits16

# Facebook DINOv2 (set --num_register_tokens 4 if using with registers)
--backbone facebook/dinov2-base
--backbone facebook/dinov2-large
```

#### Layer Indices

Choose layer indices based on model depth:

```bash
# ViT-Base (12 layers) - extract from layers 2, 5, 8, 11
--backbone_indices 2 5 8 11

# ViT-Large (24 layers) - extract from layers 5, 11, 17, 23
--backbone_indices 5 11 17 23
```

### Transfer Learning (Frozen Backbone)

Train only the segmentation head:

```bash
python train_sementic.py \
    --data_root /path/to/coco \
    --freeze_backbone \
    --lr 1e-3 \
    --epochs 50
```

### Resume Training

```bash
python train_sementic.py \
    --data_root /path/to/coco \
    --resume_from ./logs/checkpoints/last.ckpt \
    --epochs 200
```

### Debug Mode

```bash
# Quick sanity check (1 batch per epoch)
python train_sementic.py --data_root /path/to/coco --fast_dev_run

# Overfit on small subset
python train_sementic.py --data_root /path/to/coco --overfit_batches 10
```

### Memory-Efficient Training

```bash
# Gradient accumulation (effective batch = 8 * 4 = 32)
python train_sementic.py \
    --data_root /path/to/coco \
    --batch_size 8 \
    --accumulate_grad_batches 4

# Dataset caching (trades RAM for speed)
python train_sementic.py \
    --data_root /path/to/coco \
    --cache_dataset \
    --num_workers 8
```

**Note:** When using `--cache_dataset`, ensure you have enough RAM:
- COCO at 512x512: ~50GB RAM
- Smaller datasets: adjust accordingly

### HuggingFace Authentication

For gated models that require authentication:

```bash
export HF_TOKEN='your_huggingface_token_here'
python train_sementic.py \
    --data_root /path/to/coco \
    --backbone facebook/dinov2-large \
    --hf_login
```

---

## Output Structure

```
logs/
├── checkpoints/
│   ├── epoch=10-val_iou=0.6543.ckpt
│   ├── epoch=20-val_iou=0.7234.ckpt
│   └── last.ckpt
├── wandb/  # If using WandB
└── tensorboard/  # If using TensorBoard
```

---

## Troubleshooting

### Out of Memory Errors

**GPU Memory:**
- Reduce `--batch_size`
- Use `--precision 16-mixed` or `--precision bf16-mixed`
- Use `--accumulate_grad_batches` for larger effective batch
- Reduce `--img_size`
- Use smaller backbone model

**RAM (Shared Memory):**
- When using Docker with `--cache_dataset`, increase `--shm-size=50gb`
- Reduce `--num_workers`
- Disable `--cache_dataset`

### Slow Training

- Increase `--num_workers` (4-8 per GPU is typical)
- Enable `--cache_dataset` if you have enough RAM
- Use `--pin_memory` and `--persistent_workers`
- Use `--precision 16-mixed` for faster computation
- Consider using FFCV for data loading (Docker setup includes it)

### Poor Convergence

- Try different `--scheduler` (cosine usually works well)
- Add `--warmup_epochs 5-10` for better stability
- Adjust `--lr` (try 1e-4 to 6e-5)
- Check that `--backbone_indices` covers early, mid, and late layers
- Try different augmentation presets (`--augmentation medium` or `--augmentation underwater`)
- Ensure `--increase_idx` and `--fill_background` match your dataset

### CUDA Errors

**Native Installation:**
- Ensure PyTorch CUDA version matches your system CUDA
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**Docker:**
- Ensure `nvidia-docker2` is installed
- Check: `docker run --gpus all underwater-benchmark:latest nvidia-smi`

---

## Performance Tips

### Best Practices

1. **Start with medium augmentation**: `--augmentation medium`
2. **Use mixed precision**: `--precision 16-mixed`
3. **Enable warmup**: `--warmup_epochs 5`
4. **Use cosine scheduler**: `--scheduler cosine`
5. **Cache if RAM available**: `--cache_dataset` (requires 50GB+ for COCO)
6. **Optimize workers**: `--num_workers 4-8` per GPU

### Batch Size Guidelines

| GPU Memory | Batch Size (512x512) | Batch Size (256x256) |
|------------|----------------------|----------------------|
| 8GB | 2-4 | 8-16 |
| 16GB | 8-16 | 32-64 |
| 24GB | 16-32 | 64-128 |
| 40GB+ | 32-64 | 128-256 |

Use `--accumulate_grad_batches` to simulate larger batch sizes.

---

## Project Structure

```
underwaterbenchmark/
├── datamodules/          # Lightning data modules
│   └── cocodatamodule_semantic.py
├── datasets/             # PyTorch datasets
│   └── cocodataset_semantic.py
├── models/              # Model architectures
│   └── UperVitBackone.py
├── modules/             # Lightning modules
│   └── UperNetLightningModule.py
├── src/
│   └── augmentation/    # Kornia augmentations
│       └── kornia_augs.py
├── docker/              # Docker configuration
│   ├── Dockerfile
│   └── README.md
├── train_sementic.py    # Main training script
└── requirements.txt     # Python dependencies
```

---

## License

MIT

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{underwater_benchmark,
  title = {Underwater Benchmark - Semantic Segmentation},
  year = {2024},
  url = {https://github.com/yourusername/underwaterbenchmark}
}
```
