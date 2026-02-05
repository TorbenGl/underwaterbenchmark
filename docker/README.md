# Docker Setup for Underwater Benchmark

This directory contains the Docker configuration for the underwater semantic segmentation benchmark.

## Features

- NVIDIA CUDA 11.8 with cuDNN 8
- PyTorch with CUDA support
- PyTorch Lightning for training
- Transformers and timm for model architectures
- Kornia for GPU-accelerated augmentations
- FFCV for fast data loading (pre-installed and ready for future integration)
- Jupyter Lab for interactive development
- W&B for experiment tracking

## Building the Image

From the project root directory:

```bash
cd /home/tglobisch/code/underwaterbenchmark
docker build -t underwater-benchmark:latest -f docker/Dockerfile .
```

## Running the Container

### For Training (with GPU)

```bash
docker run --gpus all \
    --shm-size=50gb \
    -v /path/to/your/data:/workspace/data \
    -v /path/to/your/code:/workspace/code \
    -v /path/to/your/logs:/workspace/logs \
    -p 8888:8888 \
    underwater-benchmark:latest
```

### For Interactive Development

```bash
docker run -it --gpus all \
    --shm-size=50gb \
    -v $(pwd):/workspace/code \
    -p 8888:8888 \
    underwater-benchmark:latest \
    /bin/bash
```

## Important Notes

### Shared Memory Size

When using cached datasets with multiprocessing workers, you MUST increase shared memory:

```bash
docker run --shm-size=50gb ...
```

- For COCO-style datasets at 512x512: ~50GB recommended
- For smaller datasets: adjust accordingly
- See `datamodules/cocodatamodule_semantic.py` for memory usage estimates

### Volume Mounts

- `/workspace/code`: Mount your project code here
- `/workspace/data`: Mount your datasets here
- `/workspace/logs`: Mount for saving checkpoints and logs

### Jupyter Lab

The container starts Jupyter Lab by default on port 8888 with no password/token.

Access it at: `http://localhost:8888`

To run training instead:

```bash
docker run --gpus all \
    --shm-size=50gb \
    -v $(pwd):/workspace/code \
    underwater-benchmark:latest \
    python /workspace/code/train_sementic.py --data_root /workspace/data/coco --epochs 100
```

## Modifying Dependencies

### Adding New Python Packages

1. Edit `requirements.txt` in the project root
2. Rebuild the Docker image

### FFCV Integration

FFCV is already installed and ready to use. The installation includes:
- FFCV library from the official repository
- All required dependencies (CuPy, numba, libjpeg-turbo)
- Optimized for CUDA 11.8

To use FFCV in your code, simply import it:

```python
from ffcv.loader import Loader
from ffcv.fields import RGBImageField, IntField
```

## Troubleshooting

### Out of Memory Errors

If you see shared memory errors:
1. Increase `--shm-size` parameter
2. Reduce `num_workers` in your data loader
3. Disable dataset caching

### CUDA Errors

Ensure you have:
- NVIDIA drivers installed on host
- Docker with GPU support (`nvidia-docker2`)
- Compatible CUDA version (11.8)

Check GPU availability:

```bash
docker run --gpus all underwater-benchmark:latest nvidia-smi
```
