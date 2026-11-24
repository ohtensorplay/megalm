# syntax=docker/dockerfile:1.7
#
# Production image for the MegaLM OpenAI-compatible server.
#
# Build (TensorPlay wheels come from https://download.tensorplay.cn/simple):
#   docker build -t megalm:local .
#
# Run (requires the NVIDIA Container Toolkit):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/models:ro \
#     megalm:local --model /models/Qwen3-8B --served-model-name qwen3 --host 0.0.0.0
#
# See docs/deployment/docker.md for details.

# =============================================================================
# Version arguments (intentional source of truth for this image)
# =============================================================================
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12
ARG RUST_VERSION=1.94.0

# -----------------------------------------------------------------------------
# Stage 1: build the Python environment and the PyO3 extension with maturin.
# The devel CUDA image provides nvcc/ptxas for TensorPlay / Triton JIT kernels.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG RUST_VERSION
# TensorPlay is served from its PEP 503 index, not public PyPI. Override to a
# mirror via build-arg if needed; TENSORPLAY_SPEC can pin a specific wheel.
# Note: 1.3.0.dev* / 1.3.0a0 are pre-releases, so `>=1.3` would exclude them.
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL=https://download.tensorplay.cn/simple
# Pin the CUDA 13 wheel: the index also publishes +cpu and +maca variants of
# the same version, and uv would otherwise pick an incompatible one.
ARG TENSORPLAY_SPEC="tensorplay==1.3.0.dev20261124+cu130"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    UV_INDEX_URL=${PIP_INDEX_URL} \
    UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PATH="/root/.cargo/bin:/root/.local/bin:/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        pkg-config \
        libssl-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rustc --version && cargo --version

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /src

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY rust/ ./rust/
COPY src/ ./src/

# The Python wheel only builds the PyO3 extension; the HTTP frontend is the
# Rust server binary, so build it here as well and install it beside the venv
# executables.
RUN uv venv --python "python${PYTHON_VERSION}" /opt/venv \
    && uv pip install --pre --python /opt/venv/bin/python "${TENSORPLAY_SPEC}" \
    && uv pip install --pre --python /opt/venv/bin/python . \
    && /opt/venv/bin/python -c "import megalm, tensorplay; print('megalm OK')" \
    && cargo build --release --manifest-path rust/Cargo.toml -p megalm-server \
    && cp rust/target/release/megalm-server /opt/venv/bin/megalm-server

# -----------------------------------------------------------------------------
# Stage 2: slim runtime with the built venv, a non-root user and a safe default
# entrypoint. The devel base is kept because kernels compile JIT at runtime.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/megalm/.cache/huggingface \
    XDG_CACHE_HOME=/home/megalm/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 megalm \
    && useradd --uid 1001 --gid megalm --create-home --shell /bin/bash megalm \
    && mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}/triton" \
    && chown -R megalm:megalm /home/megalm

COPY --from=builder --chown=megalm:megalm /opt/venv /opt/venv

USER megalm
WORKDIR /home/megalm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

ENTRYPOINT ["megalm-server"]
CMD ["--help"]
