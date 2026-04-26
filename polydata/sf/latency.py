"""SF6: two-clock archive-ingestion delay distribution.

Plan 2a's `latency_distribution` returns one row per market with
percentile statistics of `ts_created - ts_received` (collector lag).
SF6 aggregates across the panel: median of each percentile, and a
distribution histogram.

Plan 2a T10 re-run finding (CLAUDE.md 2026-04-18) reframed this from
"trader latency" to "archive collector ingestion delay" — i.e., a
data-quality characteristic, not a behavioral signal.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def summarize_latency(per_market: pl.DataFrame) -> pl.DataFrame:
    """One panel-level row: median of each percentile column."""
    if per_market.height == 0:
        return pl.DataFrame([{
            "median_p50": float("nan"), "median_p90": float("nan"),
            "median_p99": float("nan"),
            "p25_p50": float("nan"), "p75_p50": float("nan"),
            "n_markets": 0,
        }], schema={
            "median_p50": pl.Float64, "median_p90": pl.Float64,
            "median_p99": pl.Float64,
            "p25_p50": pl.Float64, "p75_p50": pl.Float64,
            "n_markets": pl.UInt32,
        })
    return pl.DataFrame([{
        "median_p50": float(per_market["p50_ms"].median() or 0),
        "median_p90": float(per_market["p90_ms"].median() or 0),
        "median_p99": float(per_market["p99_ms"].median() or 0),
        "p25_p50": float(per_market["p50_ms"].quantile(0.25) or 0),
        "p75_p50": float(per_market["p50_ms"].quantile(0.75) or 0),
        "n_markets": per_market["market_id"].n_unique(),
    }], schema={
        "median_p50": pl.Float64, "median_p90": pl.Float64,
        "median_p99": pl.Float64,
        "p25_p50": pl.Float64, "p75_p50": pl.Float64,
        "n_markets": pl.UInt32,
    })


def run_sf6(
    latency_path: Path = Path("data/panel_quote/latency.parquet"),
    out_parquet: Path = Path("artifacts/sf6_latency.parquet"),
    out_png: Path = Path("artifacts/sf6_latency.png"),
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    raw = pl.read_parquet(latency_path)
    summary = summarize_latency(raw)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label, color in [
        ("p50_ms", "p50", "C0"),
        ("p90_ms", "p90", "C1"),
        ("p99_ms", "p99", "C3"),
    ]:
        if col in raw.columns:
            v = raw[col].drop_nulls().to_numpy()
            ax.hist(v, bins=50, alpha=0.45, label=label, color=color)
    ax.set_xlabel("latency (ms, log scale)")
    ax.set_ylabel("# markets")
    ax.set_xscale("log")
    ax.set_title(
        "SF6 — archive-ingestion latency (per-market percentiles, panel)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary
