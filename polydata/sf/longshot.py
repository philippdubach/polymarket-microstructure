"""SF1: longshot spread premium — median quoted spread (bps) binned by
mean mid price across panel markets. Classical prediction-market U-shape."""
from __future__ import annotations

from pathlib import Path

import polars as pl


def bin_spreads_by_price(
    per_market: pl.DataFrame, n_bins: int = 10,
) -> pl.DataFrame:
    """Given one row per market with (mean_mid, median_spread_bps),
    return one row per price bin with summary stats."""
    if per_market.height == 0:
        return pl.DataFrame(
            schema={
                "bin": pl.Int64, "bin_lo": pl.Float64, "bin_hi": pl.Float64,
                "n_markets": pl.UInt32, "median_spread_bps": pl.Float64,
                "p25_spread_bps": pl.Float64, "p75_spread_bps": pl.Float64,
            },
        )
    step = 1.0 / n_bins
    return (
        per_market.with_columns(
            pl.col("mean_mid").clip(1e-9, 1 - 1e-9).alias("mid_clipped"),
        )
        .with_columns(
            (pl.col("mid_clipped") / step).floor().cast(pl.Int64).alias("bin"),
        )
        .group_by("bin")
        .agg(
            (pl.col("bin").first() * step).alias("bin_lo"),
            (pl.col("bin").first() * step + step).alias("bin_hi"),
            pl.len().alias("n_markets").cast(pl.UInt32),
            pl.col("median_spread_bps").median().alias("median_spread_bps"),
            pl.col("median_spread_bps").quantile(0.25).alias("p25_spread_bps"),
            pl.col("median_spread_bps").quantile(0.75).alias("p75_spread_bps"),
        )
        .sort("bin")
        .select([
            "bin", "bin_lo", "bin_hi", "n_markets",
            "median_spread_bps", "p25_spread_bps", "p75_spread_bps",
        ])
    )


def run_sf1(
    spread_path: Path = Path("data/panel_quote/spread.parquet"),
    out_parquet: Path = Path("artifacts/sf1_longshot.parquet"),
    out_png: Path = Path("artifacts/sf1_longshot.png"),
    n_bins: int = 10,
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    raw = pl.read_parquet(spread_path)
    per_mkt = raw.filter(
        pl.col("mean_mid").is_not_null()
        & pl.col("median_spread_bps").is_not_null()
    ).select(["market_id", "mean_mid", "median_spread_bps"])
    binned = bin_spreads_by_price(per_mkt, n_bins=n_bins)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    binned.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = ((binned["bin_lo"] + binned["bin_hi"]) / 2).to_numpy()
    y = binned["median_spread_bps"].to_numpy()
    lo = binned["p25_spread_bps"].to_numpy()
    hi = binned["p75_spread_bps"].to_numpy()
    ax.plot(x, y, marker="o", linewidth=1.5, label="median")
    ax.fill_between(x, lo, hi, alpha=0.2, label="p25-p75")
    ax.set_xlabel("mid price")
    ax.set_ylabel("quoted spread (bps)")
    ax.set_title("SF1 — longshot spread premium")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return binned
