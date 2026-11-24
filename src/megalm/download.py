"""Download a model snapshot with the official MegaHub / HF / ModelScope SDKs.

Examples::

    python -m megalm.download hf:Qwen/Qwen3-8B
    python -m megalm.download mega://models/acme/llm@main --local-dir models/llm
    python -m megalm.download ms:Qwen/Qwen3-8B --include '*.safetensors' config.json

The command prints the resulting local checkpoint directory, which can be
passed straight to ``python -m megalm --model`` or ``megalm-server``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from megalm.model_source import (
    ModelSourceError,
    add_model_source_args,
    model_path_from_args,
)

# Backwards-compatible default for the documented Nanbeige example.
DEFAULT_REPO_ID = "Nanbeige/Nanbeige4.2-3B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help=f"local directory or Hub repo id (default: {DEFAULT_REPO_ID})",
    )
    add_model_source_args(parser)
    parser.add_argument("--quiet", action="store_true", help="do not print the path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = model_path_from_args(args.repo_id, args)
    except ModelSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
