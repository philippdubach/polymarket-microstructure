"""SF2: sparse non-uniform L2 depth.

The Plan 2a depth panel emits cumulative top-k depth per market for
k in {1, 5, 10}. We compute the L1-over-L10 ratio as a sparseness
proxy: ratio close to 1 means depth is concentrated at top of book
(sparse, prediction-market-like); ratio close to 0.1 means depth is
geometrically spread across levels (dense, equity-like). Aggregating
the per-market distribution shows whether Polymarket books are
typically thin-and-top-heavy or evenly stocked.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def depth_concentration(depth_panel: pl.DataFrame) -> pl.DataFrame:
    """Per-market L1-over-L10 concentration ratio for each side, plus a
    midpoint averaged across bid + ask."""
    if depth_panel.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8,
                "concentration_bid_l1_over_l10": pl.Float64,
                "concentration_ask_l1_over_l10": pl.Float64,
                "concentration_l1_over_l10": pl.Float64,
            },
        )
    pivot_bid = depth_panel.pivot(
        values="mean_bid_depth", index="market_id", on="k",
    ).rename({"1": "bid_1", "5": "bid_5", "10": "bid_10"})
    pivot_ask = depth_panel.pivot(
        values="mean_ask_depth", index="market_id", on="k",
    ).rename({"1": "ask_1", "5": "ask_5", "10": "ask_10"})
    joined = pivot_bid.join(pivot_ask, on="market_id", how="inner")
    return joined.with_columns(
        pl.when(pl.col("bid_10") > 0)
          .then(pl.col("bid_1") / pl.col("bid_10"))
          .otherwise(None)
          .alias("concentration_bid_l1_over_l10"),
        pl.when(pl.col("ask_10") > 0)
          .then(pl.col("ask_1") / pl.col("ask_10"))
          .otherwise(None)
          .alias("concentration_ask_l1_over_l10"),
    ).with_columns(
        ((pl.col("concentration_bid_l1_over_l10")
          + pl.col("concentration_ask_l1_over_l10")) / 2)
        .alias("concentration_l1_over_l10"),
    ).select([
        "market_id",
        "concentration_bid_l1_over_l10",
        "concentration_ask_l1_over_l10",
        "concentration_l1_over_l10",
    ])


def run_sf2(
    depth_path: Path = Path("data/panel_quote/depth.parquet"),
    out_parquet: Path = Path("artifacts/sf2_depth_profile.parquet"),
    out_png: Path = Path("artifacts/sf2_depth_profile.png"),
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    raw = pl.read_parquet(depth_path)
    summary = depth_concentration(raw)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = summary.filter(
        pl.col("concentration_l1_over_l10").is_not_null()
    )["concentration_l1_over_l10"].to_numpy()
    ax.hist(vals, bins=30, edgecolor="black")
    ax.axvline(0.1, color="green", linestyle="--",
               label="dense (geometric, L1=10% of L10)")
    ax.axvline(1.0, color="red", linestyle="--",
               label="fully top-of-book (sparse)")
    ax.set_xlabel("L1 / L10 cumulative-depth ratio")
    ax.set_ylabel("# markets")
    ax.set_title("SF2 — top-of-book depth concentration")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary
