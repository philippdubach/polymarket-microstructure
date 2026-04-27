"""T2 deliverable: copy PNG figures + render parquet tables into the
paper/figures and paper/tables directories."""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

from polydata.paper_assets import df_to_booktabs

FIGURES = [
    ("artifacts/sf1_longshot.png", "paper/figures/sf1_longshot.png"),
    ("artifacts/sf2_depth_profile.png", "paper/figures/sf2_depth_profile.png"),
    ("artifacts/sf3_blockclock.png", "paper/figures/sf3_blockclock.png"),
    ("artifacts/sf4_herfindahl.png", "paper/figures/sf4_herfindahl.png"),
    ("artifacts/sf5_category.png", "paper/figures/sf5_category.png"),
    ("artifacts/sf6_latency.png", "paper/figures/sf6_latency.png"),
    ("artifacts/sf7_wash.png", "paper/figures/sf7_wash.png"),
    ("artifacts/sf8_depth_decay.png", "paper/figures/sf8_depth_decay.png"),
    (
        "artifacts/spread_decomposition.png",
        "paper/figures/spread_decomposition.png",
    ),
]

TABLES = [
    {
        "name": "sf1_longshot",
        "parquet": "artifacts/sf1_longshot.parquet",
        "header_map": {
            "bin": "Bin", "bin_lo": "Mid lo", "bin_hi": "Mid hi",
            "n_markets": "Markets", "median_spread_bps": "Median (bps)",
            "p25_spread_bps": "p25 (bps)", "p75_spread_bps": "p75 (bps)",
        },
    },
    {
        "name": "sf3_blockclock",
        "parquet": "artifacts/sf3_blockclock.parquet",
        "header_map": {
            "n_markets": "Markets",
            "median_alignment": "Median align share",
            "p25_alignment": "p25", "p75_alignment": "p75",
            "n_above_threshold": "Above 0.15",
        },
    },
    {
        "name": "sf5_category",
        "parquet": "artifacts/sf5_category.parquet",
        "header_map": {
            "category": "Category",
            "n_markets": "Markets",
            "median": "Median half-spread (prob pp)",
            "p25": "p25", "p75": "p75",
        },
    },
    {
        "name": "sf6_latency",
        "parquet": "artifacts/sf6_latency.parquet",
        "header_map": {
            "n_markets": "Markets",
            "median_p50": "Median p50 (ms)",
            "median_p90": "Median p90 (ms)",
            "median_p99": "Median p99 (ms)",
        },
    },
    {
        "name": "sf8_depth_decay",
        "parquet": "artifacts/sf8_depth_decay.parquet",
        "header_map": {
            "n_used_in_fit": "Markets in fit",
            "slope_log_ttc": "Slope on log(ttc)",
            "se_hc3": "SE (HC3)",
            "r2": "R²",
        },
    },
    {
        "name": "spread_decomposition_summary",
        "parquet": "artifacts/spread_decomposition.parquet",
        "header_map": {
            "market_id": "Market id (truncated)",
            "eff_half": "Effective half",
            "c_transitory": "$c$ (transitory)",
            "phi_adverse_sel": "$\\varphi$ (adverse sel.)",
        },
        "head": 10,
        "truncate_id": ("market_id", 16),
    },
]


def main() -> int:
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    for src, dst in FIGURES:
        s, d = Path(src), Path(dst)
        if s.exists():
            shutil.copy(s, d)
            print(f"copied {src} -> {dst}")
        else:
            print(f"skip {src} (missing)")
    for spec in TABLES:
        p = Path(spec["parquet"])
        if not p.exists():
            print(f"skip {spec['name']} (missing {p})")
            continue
        df = pl.read_parquet(p)
        if "head" in spec:
            df = df.head(spec["head"])
        if "truncate_id" in spec:
            col, n = spec["truncate_id"]
            df = df.with_columns(
                pl.col(col).str.slice(0, n).str.replace_all("_", "\\_")
                .map_elements(lambda s: f"{s}\\ldots", return_dtype=pl.Utf8)
                .alias(col)
            )
        latex = df_to_booktabs(df, spec["header_map"])
        out = Path(f"paper/tables/{spec['name']}.tex")
        out.write_text(latex + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
