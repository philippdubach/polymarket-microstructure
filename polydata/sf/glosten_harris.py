"""Glosten-Harris (1988) spread decomposition on the top-100 panel.

Decomposition (per market):
    effective_half_spread = c (transitory) + φ (adverse selection)

Per Glosten-Harris, the realized spread (effective measured against a
post-trade reference mid) recovers the transitory component c, and the
residual is the adverse-selection component φ:
    c   = realized_half_spread (lag = 60s)
    φ   = effective_half_spread - realized_half_spread

We use Plan 3b's panel_trade_measures.parquet outputs:
- effective_spread.half_spread_prob_dw (dollar-volume weighted)
- realized_spread.realized_half_spread_dw at lag_sec = 60.

Restrict to the top-100 stratum where trade counts are high enough for
reliable estimates.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def decompose_per_market(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute c and φ per market from columns (market_id, eff_half,
    realized_half)."""
    if panel.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8,
                "c_transitory": pl.Float64,
                "phi_adverse_sel": pl.Float64,
                "eff_half": pl.Float64,
            },
        )
    return panel.with_columns(
        pl.col("realized_half").alias("c_transitory"),
        (pl.col("eff_half") - pl.col("realized_half")).alias("phi_adverse_sel"),
    ).select([
        "market_id", "eff_half", "c_transitory", "phi_adverse_sel",
    ])


def run_spread_decomposition(
    panel_trade: Path = Path("data/panel_trade_measures.parquet"),
    panel: Path = Path("data/panel.parquet"),
    out_parquet: Path = Path("artifacts/spread_decomposition.parquet"),
    out_png: Path = Path("artifacts/spread_decomposition.png"),
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    panel_df = pl.read_parquet(panel)
    top_markets = panel_df.filter(pl.col("stratum") == "top").select("market_id")
    m = pl.read_parquet(panel_trade)
    eff = m.filter(pl.col("measure") == "effective_spread").select([
        "market_id", pl.col("half_spread_prob_dw").alias("eff_half"),
    ])
    rea = m.filter(
        (pl.col("measure") == "realized_spread") & (pl.col("lag_sec") == 60)
    ).select([
        "market_id", pl.col("realized_half_spread_dw").alias("realized_half"),
    ])
    joined = (
        top_markets.join(eff, on="market_id", how="inner")
        .join(rea, on="market_id", how="inner")
        .filter(
            pl.col("eff_half").is_not_null()
            & pl.col("realized_half").is_not_null()
        )
    )
    decomp = decompose_per_market(joined)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    decomp.write_parquet(out_parquet)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    c_vals = decomp["c_transitory"].to_numpy()
    phi_vals = decomp["phi_adverse_sel"].to_numpy()
    ax1.hist(c_vals, bins=30, edgecolor="black", color="steelblue")
    ax1.set_xlabel("c (transitory, prob pp)")
    ax1.set_ylabel("# markets")
    ax1.set_title("Transitory component")
    ax2.hist(phi_vals, bins=30, edgecolor="black", color="darkorange")
    ax2.set_xlabel("φ (adverse selection, prob pp)")
    ax2.set_title("Adverse-selection component")
    fig.suptitle(
        "Glosten-Harris spread decomposition (top-100 stratum)"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return decomp
