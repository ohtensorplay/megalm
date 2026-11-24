"""Guard against a stale ``megalm_rs`` native extension in the source tree.

``src/megalm/megalm_rs*.so`` is built by maturin and is loaded directly by the
worker. Editing ``rust/src/**/*.rs`` without rebuilding leaves the old binary
in place, which makes tests and production load different parser code. This
test fails loudly in that situation. Set ``MEGALM_SKIP_SO_FRESHNESS=1`` only
for source-only jobs that deliberately do not build the extension.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_native_extension_is_newer_than_rust_sources() -> None:
    if os.environ.get("MEGALM_SKIP_SO_FRESHNESS"):
        return

    extensions = sorted((ROOT / "src/megalm").glob("megalm_rs*.so"))
    assert extensions, "megalm_rs native extension is missing; run `maturin develop --release --skip-install`"

    # maturin builds the PyO3 extension from the workspace root crate only.
    # The server / tokenizer / engine-client crates are separate binaries and
    # do not affect the .so, so they must not trigger a false positive.
    sources = [*(ROOT / "rust/src").rglob("*.rs")]
    for manifest in ("rust/Cargo.toml", "rust/Cargo.lock"):
        if (ROOT / manifest).exists():
            sources.append(ROOT / manifest)
    assert sources, "no Rust sources found"

    newest = max(sources, key=lambda path: path.stat().st_mtime)
    extension = max(extensions, key=lambda path: path.stat().st_mtime)
    assert extension.stat().st_mtime >= newest.stat().st_mtime, (
        f"{newest.relative_to(ROOT)} is newer than {extension.relative_to(ROOT)}; "
        "rebuild with `maturin develop --release --skip-install`"
    )
