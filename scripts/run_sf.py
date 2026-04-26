"""T12 unified driver: run SF1-SF8 + Glosten-Harris decomposition.

Each SF module ships a `run_sf*` function that consumes pre-computed
panel artifacts (from T2-T4) and writes parquet + png to artifacts/.
"""
from __future__ import annotations

from polydata.sf.blockclock import run_sf3
from polydata.sf.category import run_sf5
from polydata.sf.depth_decay import run_sf8
from polydata.sf.depth_profile import run_sf2
from polydata.sf.glosten_harris import run_spread_decomposition
from polydata.sf.herfindahl import run_sf4
from polydata.sf.latency import run_sf6
from polydata.sf.longshot import run_sf1
from polydata.sf.wash import run_sf7


def main() -> int:
    for label, fn in [
        ("SF1 longshot", run_sf1),
        ("SF2 depth profile", run_sf2),
        ("SF3 blockclock", run_sf3),
        ("SF4 herfindahl", run_sf4),
        ("SF5 category", run_sf5),
        ("SF6 latency", run_sf6),
        ("SF7 wash", run_sf7),
        ("SF8 depth decay", run_sf8),
        ("Glosten-Harris", run_spread_decomposition),
    ]:
        print(f"\n=== {label} ===")
        out = fn()
        h = out.height if hasattr(out, "height") else "?"
        print(f"rows: {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
