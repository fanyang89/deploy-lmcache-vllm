# Base image as specified
FROM nvcr.io/nvidia/pytorch:25.05-py3

# --- Build Arguments ---
# Use --build-arg to set this. Examples:
# Hopper (H100/H200):  CUDA_ARCHS="9.0"
# Ampere (A100):       CUDA_ARCHS="8.0;8.6;8.9"
# Ada Lovelace (4090): CUDA_ARCHS="8.9" (compiles as Ampere, runs on Ada)
# https://developer.nvidia.com/cuda-gpus
ARG CUDA_ARCHS="12.0"

# Corresponding format for cmake/make (e.g., "90" for Hopper, "86" for A100)
ARG CMAKE_CUDA_ARCHS="native"

# Set shell to bash for commands like $(nproc)
SHELL ["/bin/bash", "-c"]

# --- Environment Variables for Compilation ---
# This block corresponds to your 'export' commands.
ENV \
    # Core PyTorch Components
    USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_MKLDNN=1 \
    USE_XNNPACK=1 \
    USE_DISTRIBUTED=1 \
    USE_NCCL=1 \
    USE_SYSTEM_NCCL=1 \
    \
    # Inductor + AOT stack
    USE_INDUCTOR=1 \
    USE_AOT_RUNTIME=1 \
    USE_TRITON=1 \
    USE_FUNCTORCH=1 \
    TORCHINDUCTOR_DEV=1 \
    \
    # Misc Build Flags
    BUILD_CAFFE2=0 \
    BUILD_TEST=0 \
    USE_FLASH_ATTENTION=1 \
    USE_TENSORRT=0 \
    \
    # Build Metadata & Options
    PYTORCH_BUILD_NUMBER=1 \
    MAX_JOBS=$(nproc) \
    FLASHINFER_ENABLE_AOT=1 \
    CCACHE_DIR=/root/.ccache \
    \
    # --- CRITICAL: Set CUDA Architectures from ARG ---
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" \
    FLASH_ATTN_CUDA_ARCHS="${CMAKE_CUDA_ARCHS}"

# Set working directory for all our builds
WORKDIR /workspace

RUN pip install --upgrade "packaging>=24.2"

# Install essential build dependencies
# git, cmake, and ninja-build are usually in the base image, but we ensure they are present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    ccache \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install build-time and runtime dependencies in fewer layers
RUN pip install --no-cache-dir \
    triton \
    "numpy<2.3" \
    aiohttp protobuf click rich starlette distro httpx jiter pydantic \
    safetensors uvloop uvicorn python-multipart accelerate msgpack llvmlite

# ------------------------------------------------------------------------------
# 1. Compile and Install PyTorch
# ------------------------------------------------------------------------------
#RUN git clone --recursive https://github.com/pytorch/pytorch.git
#WORKDIR /workspace/pytorch

#RUN pip install --no-cache-dir -r requirements.txt

# Build PyTorch. Using --mount for ccache to speed up rebuilds.
# Note: Changed 'install install' to the standard 'install'.
#RUN --mount=type=cache,target=/root/.ccache \
#    MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) python setup.py --verbose install

# ------------------------------------------------------------------------------
# 2. Compile and Install FlashInfer
# ------------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git
WORKDIR /workspace/flashinfer

# Build FlashInfer using CMake and Ninja
#RUN  python3 -m flashinfer.aot --out-dir=/root/.cache/flashinfer --f16-dtype bfloat16  --f8-dtype float8_e4m3fn
#RUN  python3 -m flashinfer.aot --out-dir=/root/.cache/flashinfer
RUN --mount=type=cache,target=/root/.ccache \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHS}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -GNinja && \
    ninja -j$(nproc)

# Install FlashInfer via pip in editable mode
RUN pip install -e . --no-build-isolation --no-deps --verbose

# ------------------------------------------------------------------------------
# 3. Compile and Install bitsandbytes
# ------------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
WORKDIR /workspace/bitsandbytes

# Build bitsandbytes
RUN --mount=type=cache,target=/root/.ccache \
    cmake . -DCOMPUTE_CAPABILITY=${CMAKE_CUDA_ARCHS} -DCOMPUTE_BACKEND=cuda -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Install bitsandbytes via pip in editable mode
RUN --mount=type=cache,target=/root/.ccache \
    pip install -e . --no-build-isolation --no-deps --verbose

WORKDIR /workspace

# Set vLLM-specific environment variables
ENV USE_CUDNN=1 \
    USE_CUSPARSELT=1 \
    NCCL_CUMEM_ENABLE=1

# Clone your specified vLLM fork
RUN git clone https://github.com/vllm-project/vllm vllm/
WORKDIR /workspace/vllm/

# vLLM build process
RUN python use_existing_torch.py
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -r requirements/common.txt
RUN pip install --no-cache-dir setuptools_scm

# Build vLLM in development mode. Using --mount for ccache.
RUN --mount=type=cache,target=/root/.ccache \
    #MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) python setup.py --verbose develop
    MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)  pip install . --no-deps --no-build-isolation --verbose

# ------------------------------------------------------------------------------
# Finalization
# ------------------------------------------------------------------------------
# Clean up build directories to reduce final image size
RUN rm -rf /workspace

# Set the final working directory
WORKDIR /
