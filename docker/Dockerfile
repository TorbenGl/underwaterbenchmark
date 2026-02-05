# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# NOTE: When using cached datasets with multiprocessing workers, 
# run with: docker run --shm-size=50gb (adjust based on dataset size)
# See datamodules/cocodatamodule_semantic.py for memory usage estimates

# Set working directory
WORKDIR /workspace

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    ffmpeg \
    pkg-config \
    libjpeg-turbo8-dev \
    libturbojpeg0-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CuPy for CUDA 11.8
RUN pip install --no-cache-dir cupy-cuda11x

# Install numba
RUN pip install --no-cache-dir numba

# Copy and install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Clone and install FFCV
RUN git clone https://github.com/libffcv/ffcv.git /workspace/ffcv && \
    cd /workspace/ffcv && \
    pip install --no-cache-dir .

# Expose Jupyter port
EXPOSE 8888

# Create directories for code, logs, and datasets
RUN mkdir -p /workspace/code /workspace/logs /workspace/data

# Set Jupyter configuration to allow access from all IPs
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# Start Jupyter Lab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]