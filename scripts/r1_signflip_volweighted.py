"""R1 — Round 1 peer-review required revision #1.

Recompute §7.2 sign-flip rates with on-chain dollar volume weights and
report the cumulative dollar-volume share of the comparable subset.

Reads artifacts/measures_compare_top100.parquet (already produced by
scripts/compare_strict_vs_onchain_top100.py for the 7-day window
2026-03-07 -> 2026-03-14). Outputs a tidy CSV under
artifacts/r1_signflip_volweighted.csv plus a stdout summary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

ART = Path("artifacts/measures_compare_top100.parquet")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return centre - half, centre + half


def vw_ci_bootstrap(
    flips: np.ndarray, w: np.ndarray, n_boot: int = 10000, seed: int = 20260427,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(flips)
    point = float((flips * w).sum() / w.sum())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = (flips[idx] * w[idx]).sum() / w[idx].sum()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _market_onchain_volume(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(
            (pl.col("measure") == "effective_spread")
            & (pl.col("trade_source") == "onchain")
            & pl.col("dollar_volume").is_finite()
        )
        .select(
            pl.col("market_id"),
            pl.col("dollar_volume").alias("dollar_volume_onchain"),
        )
        .unique(subset="market_id")
    )


def _flip_subset(
    df: pl.DataFrame, measure: str, value_col: str,
) -> pl.DataFrame:
    sub = df.filter(pl.col("measure") == measure)
    pivot = sub.pivot(
        values=[value_col, "n_trades"],
        index="market_id",
        on="trade_source",
    )
    val_strict = f"{value_col}_strict"
    val_onchain = f"{value_col}_onchain"
    if val_strict not in pivot.columns or val_onchain not in pivot.columns:
        return pl.DataFrame()
    vol = _market_onchain_volume(df)
    keep = (
        pivot.filter(
            pl.col(val_strict).is_finite() & pl.col(val_onchain).is_finite()
        )
        .join(vol, on="market_id", how="left")
    )
    return keep.with_columns(
        ((pl.col(val_strict).sign() != pl.col(val_onchain).sign()).cast(pl.Int64))
        .alias("flip"),
    )


def summarise(name: str, comp: pl.DataFrame, total_volume_top100: float) -> dict:
    n = comp.height
    k = int(comp["flip"].sum())
    p_unw = k / n if n else float("nan")
    lo_unw, hi_unw = wilson(k, n)
    flips = comp["flip"].to_numpy().astype(float)
    w_raw = comp["dollar_volume_onchain"].to_numpy().astype(float)
    w = np.where(np.isfinite(w_raw), w_raw, 0.0)
    if w.sum() <= 0:
        p_vw, lo_vw, hi_vw = float("nan"), float("nan"), float("nan")
    else:
        p_vw, lo_vw, hi_vw = vw_ci_bootstrap(flips, w)
    subset_vol = float(w.sum())
    share = subset_vol / total_volume_top100 if total_volume_top100 > 0 else float("nan")
    return {
        "measure": name,
        "n": n, "k_flip": k,
        "p_unweighted": p_unw, "ci_lo_unw": lo_unw, "ci_hi_unw": hi_unw,
        "p_volweighted": p_vw, "ci_lo_vw": lo_vw, "ci_hi_vw": hi_vw,
        "subset_dollar_volume": subset_vol,
        "subset_share_of_top100": share,
    }


def main() -> int:
    df = pl.read_parquet(ART)
    onchain_all = df.filter(
        (pl.col("trade_source") == "onchain")
        & (pl.col("dollar_volume").is_not_null())
    )
    total_volume_top100 = float(
        onchain_all.unique(subset="market_id")["dollar_volume"].sum()
    )
    print(
        f"top-100 on-chain dollar volume (7d): "
        f"${total_volume_top100/1e6:.2f}M",
        flush=True,
    )

    rows: list[dict] = []
    eff = _flip_subset(df, "effective_spread", "half_spread_prob_dw")
    rows.append(summarise("effective_spread", eff, total_volume_top100))
    kyle = _flip_subset(df, "kyle_lambda", "kyle_lambda_logodds")
    rows.append(summarise("kyle_lambda", kyle, total_volume_top100))

    out = pl.DataFrame(rows)
    out_path = Path("artifacts/r1_signflip_volweighted.csv")
    out.write_csv(out_path)
    print()
    for r in rows:
        print(
            f"{r['measure']:>20s}  n={r['n']:>3d}  k={r['k_flip']:>3d}  "
            f"unw={r['p_unweighted']:.3f}  Wilson95=[{r['ci_lo_unw']:.3f},"
            f"{r['ci_hi_unw']:.3f}]  vw={r['p_volweighted']:.3f}  "
            f"boot95=[{r['ci_lo_vw']:.3f},{r['ci_hi_vw']:.3f}]  "
            f"subset_share={r['subset_share_of_top100']:.3f}",
            flush=True,
        )
    eff = _flip_subset(df, "effective_spread", "half_spread_prob_dw")
    kyle = _flip_subset(df, "kyle_lambda", "kyle_lambda_logodds")
    eff.sort("dollar_volume_onchain", descending=True, nulls_last=True).write_csv(
        "artifacts/r1_signflip_effspread_per_market.csv"
    )
    kyle.sort("dollar_volume_onchain", descending=True, nulls_last=True).write_csv(
        "artifacts/r1_signflip_kyle_per_market.csv"
    )
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
