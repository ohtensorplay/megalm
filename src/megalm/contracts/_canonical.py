from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def canonical_value(value: Any, *, omit: frozenset[str] = frozenset()) -> Any:
    """Convert contract data into a deterministic JSON-compatible value."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_value(getattr(value, field.name), omit=omit)
            for field in fields(value)
            if field.name not in omit
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item, omit=omit)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_value(item, omit=omit) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [canonical_value(item, omit=omit) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "byte_length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"contract value is not canonically serializable: {type(value)!r}")


def canonical_json(value: Any, *, omit: frozenset[str] = frozenset()) -> str:
    return json.dumps(
        canonical_value(value, omit=omit),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_digest(value: Any, *, omit: frozenset[str] = frozenset()) -> str:
    payload = canonical_json(value, omit=omit).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

