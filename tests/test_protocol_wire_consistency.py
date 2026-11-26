"""Guard the MGLM v6 wire contract across Python, PyO3 and the Rust protocol crate.

The gateway host path moved to Rust, but the GPU worker still speaks MGLM v6
through ``megalm.gateway.protocol``. A change to the frame layout in only one
of the three definitions would fail at runtime in the most expensive possible
way, so the constants and the actual encoded bytes are asserted here.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from megalm import megalm_rs

ROOT = Path(__file__).resolve().parents[1]
PY_PROTOCOL = (ROOT / "src/megalm/gateway/protocol.py").read_text()
RS_PYO3 = (ROOT / "rust/src/frames.rs").read_text()
RS_PROTOCOL = (ROOT / "rust/crates/protocol/src/lib.rs").read_text()


def _match(pattern: str, text: str) -> str:
    found = re.search(pattern, text)
    assert found is not None, f"pattern not found: {pattern}"
    return found.group(1)


def _product(expr: str) -> int:
    """Evaluate only integer products/powers from a source constant."""
    result = 1
    for factor in expr.replace("**", "^pow^").split("*"):
        factor = factor.strip()
        if "^pow^" in factor:
            base, power = factor.split("^pow^")
            result *= int(base) ** int(power)
        else:
            result *= int(factor)
    return result


def test_frame_constants_agree_across_all_three_definitions() -> None:
    magic_py = _match(r'_MAGIC\s*=\s*b"([^"]+)"', PY_PROTOCOL).encode()
    magic_pyo3 = _match(r'FRAME_MAGIC:\s*&\[u8;\s*4\]\s*=\s*b"([^"]+)"', RS_PYO3).encode()
    magic_rs = _match(r'pub const MAGIC:\s*\[u8;\s*4\]\s*=\s*\*b"([^"]+)"', RS_PROTOCOL).encode()
    assert magic_py == magic_pyo3 == magic_rs == b"MGLM"

    version_py = int(_match(r"_VERSION\s*=\s*(\d+)", PY_PROTOCOL))
    version_pyo3 = int(_match(r"FRAME_VERSION:\s*u8\s*=\s*(\d+)", RS_PYO3))
    version_rs = int(_match(r"pub const VERSION:\s*u8\s*=\s*(\d+)", RS_PROTOCOL))
    assert version_py == version_pyo3 == version_rs == 6

    header_py = _match(r'_HEADER\s*=\s*struct\.Struct\("([^"]+)"\)', PY_PROTOCOL)
    assert struct.calcsize(header_py) == 10
    header_pyo3 = int(_match(r"FRAME_HEADER_SIZE:\s*usize\s*=\s*(\d+)", RS_PYO3))
    header_rs = int(_match(r"pub const HEADER_LEN:\s*usize\s*=\s*(\d+)", RS_PROTOCOL))
    assert struct.calcsize(header_py) == header_pyo3 == header_rs == 10

    max_py = _match(r"_MAX_FRAME_BYTES\s*=\s*([0-9* ]+)", PY_PROTOCOL).strip()
    max_pyo3 = _match(r"MAX_FRAME_BYTES:\s*usize\s*=\s*([0-9 *]+);", RS_PYO3).strip()
    max_rs = _match(r"pub const MAX_BODY_BYTES:\s*usize\s*=\s*([0-9 *]+);", RS_PROTOCOL).strip()
    assert _product(max_py) == _product(max_pyo3) == _product(max_rs) == 64 * 1024**2

    type_min = int(_match(r"FRAME_TYPE_MIN:\s*u8\s*=\s*(\d+)", RS_PYO3))
    type_max = int(_match(r"FRAME_TYPE_MAX:\s*u8\s*=\s*(\d+)", RS_PYO3))
    py_types = {
        name: int(value)
        for name, value in re.findall(r"^\s*([A-Z_]+)\s*=\s*(\d+)\s*$", PY_PROTOCOL, re.M)
        if name in {
            "READY", "SUBMIT_BATCH", "SUBMIT_RESULT", "CANCEL", "INSPECT",
            "SNAPSHOT", "OUTPUTS", "STOP", "FAILURE", "SUBMIT_MULTIMODAL",
            "SUBMIT_MULTIMODAL_BATCH", "SUBMIT_EMBEDDINGS", "EMBEDDING_RESULT",
        }
    }
    assert len(py_types) == 13
    assert min(py_types.values()) == type_min
    assert max(py_types.values()) == type_max
    assert sorted(py_types.values()) == list(range(type_min, type_max + 1))

    rs_enum = _match(
        r"pub enum FrameType\s*\{([^}]+)\}",
        RS_PROTOCOL,
    )
    rs_types = {
        name: int(value)
        for name, value in re.findall(r"([A-Za-z0-9_]+)\s*=\s*(\d+)", rs_enum)
    }
    assert len(rs_types) == 13
    for name, value in py_types.items():
        rust_name = "".join(part.capitalize() for part in name.lower().split("_"))
        assert rs_types[rust_name] == value, f"FrameType.{rust_name} diverged"


def test_pyo3_encoder_matches_struct_pack_for_every_frame_type() -> None:
    body = '{"hello":"world"}'
    for frame_type in range(1, 14):
        expected = struct.pack("!4sBBI", b"MGLM", 6, frame_type, len(body)) + body.encode()
        encoded = megalm_rs.encode_replica_frame(frame_type, body)
        assert bytes(encoded) == expected
        kind, size = megalm_rs.decode_replica_frame_header(encoded[:10])
        assert (kind, size) == (frame_type, len(body))
