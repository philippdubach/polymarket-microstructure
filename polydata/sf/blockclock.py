"""SF3: Polygon block-clock effects.

Plan 2a's `block_alignment` measure reports per-market the share of
PriceChange events that fall within ±100ms of a Polygon block boundary
(2s period). Under a uniform-timing null, expected share = 200/2000 =
0.10. SF3 aggregates the per-market shares and reports how much of the
panel sits above a threshold (default 0.15 = 50% premium over null).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from scipy.stats import binomtest


def alignment_summary(
    clock_panel: pl.DataFrame,
    null_share: float = 0.10,
    threshold_above: float = 0.15,
) -> pl.DataFrame:
    """One-row summary across the panel."""
    if clock_panel.height == 0:
        return pl.DataFrame([{
            "n_markets": 0,
            "n_above_threshold": 0,
            "share_above_threshold": 0.0,
            "median_alignment": float("nan"),
            "p25_alignment": float("nan"),
            "p75_alignment": float("nan"),
            "null_share": null_share,
            "threshold": threshold_above,
        }], schema={
            "n_markets": pl.UInt32, "n_above_threshold": pl.UInt32,
            "share_above_threshold": pl.Float64,
            "median_alignment": pl.Float64,
            "p25_alignment": pl.Float64, "p75_alignment": pl.Float64,
            "null_share": pl.Float64, "threshold": pl.Float64,
        })
    n = clock_panel.height
    above = int(
        (clock_panel["aligned_share"] >= threshold_above).sum()
    )
    return pl.DataFrame([{
        "n_markets": n,
        "n_above_threshold": above,
        "share_above_threshold": above / n,
        "median_alignment": float(clock_panel["aligned_share"].median() or 0),
        "p25_alignment": float(clock_panel["aligned_share"].quantile(0.25) or 0),
        "p75_alignment": float(clock_panel["aligned_share"].quantile(0.75) or 0),
        "null_share": null_share,
        "threshold": threshold_above,
    }], schema={
        "n_markets": pl.UInt32, "n_above_threshold": pl.UInt32,
        "share_above_threshold": pl.Float64,
        "median_alignment": pl.Float64,
        "p25_alignment": pl.Float64, "p75_alignment": pl.Float64,
        "null_share": pl.Float64, "threshold": pl.Float64,
    })


def per_market_chisquare(
    clock_panel: pl.DataFrame,
    null_share: float = 0.10,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """For each market, run a two-sided binomial test of the null
    H0: share_aligned = null_share. Returns p-values + reject flags."""
    if clock_panel.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8, "p_value": pl.Float64,
                "reject_at_alpha": pl.Boolean,
            },
        )
    rows: list[dict] = []
    for r in clock_panel.iter_rows(named=True):
        n = int(r["n"])
        k = int(r["aligned"])
        if n == 0:
            rows.append({
                "market_id": r["market_id"],
                "p_value": float("nan"),
                "reject_at_alpha": False,
            })
            continue
        p = binomtest(k, n, p=null_share, alternative="two-sided").pvalue
        rows.append({
            "market_id": r["market_id"],
            "p_value": float(p),
            "reject_at_alpha": bool(p < alpha),
        })
    return pl.DataFrame(rows)


def run_sf3(
    clock_path: Path = Path("data/panel_quote/clock.parquet"),
    out_parquet: Path = Path("artifacts/sf3_blockclock.parquet"),
    out_png: Path = Path("artifacts/sf3_blockclock.png"),
    null_share: float = 0.10,
    alpha: float = 0.05,
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    raw = pl.read_parquet(clock_path)
    summary = alignment_summary(raw, null_share=null_share)
    chi = per_market_chisquare(raw, null_share=null_share, alpha=alpha)
    n_reject = int(chi["reject_at_alpha"].sum())
    n_total = int(chi.height)
    summary = summary.with_columns(
        pl.lit(n_reject).alias("n_markets_reject_null").cast(pl.UInt32),
        pl.lit(n_reject / n_total if n_total else 0.0)
            .alias("share_markets_reject_null"),
        pl.lit(alpha).alias("test_alpha"),
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = raw["aligned_share"].to_numpy()
    ax.hist(vals, bins=40, edgecolor="black")
    ax.axvline(
        0.10, color="red", linestyle="--",
        label="null (0.10)",
    )
    ax.set_xlabel("share of PriceChange events within ±100ms of block boundary")
    ax.set_ylabel("# markets")
    ax.set_title("SF3 — Polygon block-clock alignment (panel)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary
