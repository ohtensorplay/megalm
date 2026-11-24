# Docker deployment

The root `Dockerfile` builds a production image for the OpenAI-compatible
server. It is a two-stage build:

| Stage | Purpose |
| --- | --- |
| `builder` | `nvidia/cuda:<CUDA>-devel-ubuntu<VERSION>` with Python 3.12, Rust and `uv`; installs TensorPlay and builds the PyO3/maturin extension. |
| `runtime` | Same CUDA devel base (kernels compile JIT at runtime), the built `/opt/venv`, a non-root `megalm` user (UID 1001) and `megalm-server` as the entrypoint. |

Model weights are **not** baked into the image; mount them at runtime.

## Prerequisites

- Linux with an NVIDIA Ampere or newer GPU and a compatible driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/installation-guide.html) so `--gpus all` works
- Docker with BuildKit (`docker buildx version`)

## Build

TensorPlay wheels are served from its PEP 503 index
`https://download.tensorplay.cn/simple`, which is the Dockerfile default.
A plain build works when that index is reachable:

```bash
docker build -t megalm:local .
```

Override `PIP_EXTRA_INDEX_URL` only when using a mirror. Use `TENSORPLAY_SPEC`
to pin a specific wheel (note that 1.3.0.dev* / 1.3.0a0 are pre-releases, so
the default constraint is `>=1.3.0.dev0,<2` and the build passes `--pre`):

```bash
docker build \
  --build-arg TENSORPLAY_SPEC="tensorplay==1.3.0.dev20261124+cu130" \
  -t megalm:local .
```

Supported build arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `CUDA_VERSION` | `13.0.0` | CUDA toolkit version of the base image (matches the `+cu130` wheel) |
| `TENSORPLAY_SPEC` | `tensorplay==1.3.0.dev20261124+cu130` | CUDA 13 TensorPlay wheel; override both args together for another CUDA version |
| `UBUNTU_VERSION` | `24.04` | Ubuntu release of the base image |
| `PYTHON_VERSION` | `3.12` | Python interpreter installed from apt |
| `RUST_VERSION` | `1.94.0` | Rust toolchain used by maturin |
| `PIP_INDEX_URL` | _(unset)_ | Primary PyPI mirror (also exported to uv) |
| `PIP_EXTRA_INDEX_URL` | `https://download.tensorplay.cn/simple` | PEP 503 index serving TensorPlay wheels |
| `TENSORPLAY_SPEC` | `tensorplay>=1.3.0.dev0,<2` | TensorPlay requirement installed before the project |

To verify only the Dockerfile syntax (no GPU or index access needed):

```bash
docker build --check .
```

CI runs both this check and a real image build in
[`.github/workflows/docker.yml`](../../.github/workflows/docker.yml) on pull
requests that touch the image. The full build needs access to the TensorPlay
index; if the index's nightly entry points at a removed GitHub asset, the
build fails loudly instead of shipping an image without a GPU runtime.

## Run

```bash
docker run --rm --gpus all --shm-size=2g -p 8000:8000 \
  -v /path/to/models:/models:ro \
  megalm:local /models/Qwen3-8B \
  --served-model-name qwen3 \
  --host 0.0.0.0 --port 8000
```

Everything after the image name is passed to `megalm-server`; run
`docker run --rm megalm:local --help` to list the serving flags.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}]}'
```

The container exposes port `8000`, runs as the non-root `megalm` user and has a
healthcheck on `/health`. Caches (Hugging Face, Triton) live under
`/home/megalm/.cache`; to persist or share them, mount a writable volume there.

For offline inference, override the entrypoint:

```bash
docker run --rm --gpus all -v /path/to/models:/models:ro \
  --entrypoint python megalm:local -m megalm \
  --model /models/Qwen3-8B --prompt "Hello" --max-new-tokens 16
```
