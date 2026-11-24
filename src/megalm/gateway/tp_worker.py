"""Tensor-parallel follower rank process.

Followers own a shard of the model, the KV cache and a CUDA Graph runner, but
they never talk to the gateway. Rank0 broadcasts control commands and a step
tick through the TP process group; every rank then enters the same model
forward sequence so the sealed collective sites stay in lockstep.
"""

from __future__ import annotations

import argparse
import base64
import logging
from typing import Any

import orjson

from megalm.runtime.engine import Engine, EngineConfig

_LOGGER = logging.getLogger("megalm.tp_worker")


def _decode_config(payload: str) -> EngineConfig:
    return EngineConfig(**orjson.loads(base64.urlsafe_b64decode(payload)))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="megalm-tp-worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    config = _decode_config(args.config)
    if config.tp_rank == 0:
        raise SystemExit("tp_worker must be launched with a non-zero rank")
    engine = Engine(config, emit_engine_events=False)
    try:
        while True:
            command: dict[str, Any] | None = engine.tp_group.broadcast_object(None)
            if command is None:
                continue
            kind = command.get("kind")
            if kind == "stop":
                break
            if kind == "step":
                pending = engine.begin_step()
                engine.finish_step_compact(pending)
                continue
            engine.tp_apply_command(command)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:  # pragma: no cover - process boundary
        _LOGGER.exception("tensor-parallel follower failed: %s", exc)
        return 1
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
