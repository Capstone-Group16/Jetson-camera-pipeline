# Use the official DeepStream image as the foundation
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# 1. Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory to where the volume is mounted
WORKDIR /workspace

# 3. Install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 4. Initialize Git LFS
RUN git lfs install

# Base Image: nvcr.io/nvidia/deepstream:7.1-triton-multiarch (for laptop) and 7.1-samples-l4t (for Nano).