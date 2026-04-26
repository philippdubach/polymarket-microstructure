"""T2 driver: compute market stats, apply pre-registered selection rule,
write data/panel.parquet + emit SHA-256 for the pre-registration doc."""
from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from polydata.panel.stratify import build_panel


def main() -> int:
    panel = build_panel()
    out = Path("data/panel.parquet")
    panel.write_parquet(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {panel.height:,} rows to {out}")
    print(
        "  top stratum:    "
        f"{panel.filter(pl.col('stratum') == 'top').height}"
    )
    print(
        "  random stratum: "
        f"{panel.filter(pl.col('stratum') == 'random').height}"
    )
    print(f"sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
