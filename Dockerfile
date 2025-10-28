# MIT License

# Copyright (c) 2020 FT Autonomous Team One

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Base image with CUDA support for GPU-accelerated training
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
    python3-dev \
    python3-pip \
    python3-tk \
    nano \
    git \
    unzip \
    wget \
    build-essential \
    autoconf \
    libtool \
    cmake \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install OpenGL and X11 dependencies for rendering
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install core scientific computing packages with version constraints
RUN pip3 install --no-cache-dir \
    "numpy>=1.21.0,<1.25.0" \
    "scipy>=1.7.3" \
    "numba>=0.56.4,<0.59.0"

# Install Stable Baselines3 and related RL packages
RUN pip3 install --no-cache-dir \
    "stable-baselines3[extra]>=2.0.0" \
    sb3-contrib \
    tensorboard

# Install Gymnasium and Gym compatibility
RUN pip3 install --no-cache-dir \
    "gymnasium>=0.28.1" \
    gymnasium-robotics \
    shimmy[gym-v21]

# Install visualization and rendering packages
RUN pip3 install --no-cache-dir \
    "Pillow>=9.0.1" \
    "pyglet<1.6" \
    pyopengl \
    opencv-python \
    matplotlib

# Install utilities and other dependencies
RUN pip3 install --no-cache-dir \
    "pyyaml>=5.3.1" \
    shapely \
    wandb \
    pylint \
    autopep8 \
    pytest

# Set environment variables
ENV HOME=/home/F1Tenth-RL
ENV PYTHONPATH="${PYTHONPATH}:/home/F1Tenth-RL:/home/F1Tenth-RL/f1tenth_gym_ros"

# Create working directory
WORKDIR /home/F1Tenth-RL

# Copy project files
COPY . /home/F1Tenth-RL/

# Install F1Tenth Gym package
RUN cd /home/F1Tenth-RL/f1tenth_gym_ros && \
    pip3 install --no-cache-dir -e .

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python3 -c "import stable_baselines3; print(f'Stable-Baselines3: {stable_baselines3.__version__}')" && \
    python3 -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')" && \
    python3 -c "import gym; print(f'Gym: {gym.__version__}')"

# Open terminal when container starts
ENTRYPOINT ["/bin/bash"]
