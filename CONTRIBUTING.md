# Contributing to MegaLM

Thanks for taking the time to improve MegaLM. This document describes the
development setup, repository layout, checks, and pull-request process.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
For security vulnerabilities, please follow [SECURITY.md](SECURITY.md) instead
of opening a public issue.

## Development setup

Requirements:

- Python 3.12 or newer
- A stable Rust toolchain (the Python extension is a PyO3/maturin module)
- Linux with CUDA-capable hardware for the GPU tests and benchmarks; CPU-only
  development and the majority of the unit tests work without a GPU

```bash
git clone https://github.com/ohtensorplay/megalm.git
cd megalm
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The editable install builds the Rust extension through maturin. To work on the
Rust side directly, use `cargo` in the `rust/` subproject:

```bash
cargo check --manifest-path rust/Cargo.toml --locked
cargo test  --manifest-path rust/Cargo.toml --locked
```

Optional extras: `.[structured]` enables engine-level constrained decoding
(`xgrammar`), and `.[otel]` enables OTLP metrics and traces.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/megalm/adapters/` | Model adapters (Qwen3/3.5, Qwen2.5-Omni, Inkling-2, OXR, Nanbeige, ...) |
| `src/megalm/contracts/` | Canonical request / plan / cache / LoRA data contracts |
| `src/megalm/kernels/` | TensorPlay and Triton kernels plus their fallback paths |
| `src/megalm/runtime/` | Engine, scheduler, KV cache, sampler, multimodal pipelines |
| `src/megalm/gateway/` | HTTP serving layer, OpenAI-compatible APIs, protocol parsing |
| `src/megalm/models/`, `loading/` | Model definitions and weight / tokenizer loading |
| `rust/` | PyO3 extension: parser, radix cache, SSE and metrics helpers |
| `tests/` | Unit tests (`test_*.py`) and standalone benchmarks (`bench_*.py`) |
| `docs/` | Quickstart, design notes, model notes and benchmark artifacts |

Keep new code in the matching layer; in particular, public request/response
shapes belong in `contracts/`, and kernel-level optimizations belong in
`kernels/` with a documented fallback in `fallback.py`.

## Checks

Run these before opening a pull request — the same gates run in CI:

```bash
ruff check src tests                                  # syntax / undefined-name gate
python -m pytest tests -q                             # CPU tests; GPU tests skip without CUDA
cargo check --manifest-path rust/Cargo.toml --locked
```

Tests whose names end in `_cuda` and tests guarded by `requires CUDA` skip
automatically when no device is available. If you have a GPU, run them with
`python -m pytest tests -q -k cuda`. The `tests/bench_*.py` files are manual
benchmarks and are not collected by pytest.

To install the repository hygiene hooks:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Docker image

The root `Dockerfile` is a two-stage builder/runtime image. Validate it without
a GPU or a private package index with `docker build --check .` (the same check
runs in `.github/workflows/docker.yml`). Build the full image by pointing the
build at your TensorPlay index:

```bash
docker build \
  --build-arg PIP_EXTRA_INDEX_URL=https://download.tensorplay.cn/simple \
  -t megalm:local .
```

See [Docker deployment](docs/deployment/docker.md) for runtime instructions.

## Pull requests

- Use the default PR template and keep the title/description accurate — they
  become the squash-commit message used when bisecting.
- Keep PRs focused. Separate features, fixes, and refactors into separate PRs.
- Add or update tests for behavioral changes; update the relevant page in
  `docs/` when behavior or flags change.
- For performance work, include before/after numbers, the GPU and problem
  sizes, and the benchmark command so the result is reproducible.
- Add an entry under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for
  user-visible changes.
- AI-assisted contributions are welcome, but authors are expected to
  understand and be able to explain their changes during review.

CI runs on pull requests into `main`. Draft PRs are supported; mark a PR ready
for review when you want the full check suite and reviewer attention.

## Releases

Versions follow `major.minor.patch` in `pyproject.toml`. Maintainers publish by
pushing a `v*` tag, which triggers the build-and-publish workflow for the
maturin wheels and source distribution. Update `CHANGELOG.md` first.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
