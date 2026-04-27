"""R2 — Round 1 peer-review required revision #2.

Compute a clustered (market-level) bootstrap 95% CI on the panel-level
sign-agreement mean reported in §7.1. Two aggregates are reported:

1. Simple panel mean across (market, window) cells.
2. Volume-weighted panel mean using on-chain matched-bucket counts.

The cluster unit is `market_id`; each bootstrap iteration resamples
markets with replacement and takes all their valid cells.

Reads `artifacts/sign_agreement_robustness.parquet`.
Writes `artifacts/r2_panel_signagreement_ci.txt` with the summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ART = Path("artifacts/sign_agreement_robustness.parquet")
OUT = Path("artifacts/r2_panel_signagreement_ci.txt")
N_BOOT = 10_000
SEED = 20260427


def main() -> int:
    df = pl.read_parquet(ART)
    valid = df.filter(
        pl.col("sign_agreement").is_not_null()
        & pl.col("sign_agreement").is_not_nan()
    )
    n_cells = valid.height
    n_markets = valid["market_id"].n_unique()
    sa_simple = float(valid["sign_agreement"].mean())
    n_matched = valid["n_matched"].cast(pl.Float64)
    sa_vw = float(
        (valid["sign_agreement"] * n_matched).sum() / n_matched.sum()
    )
    total_matched = int(n_matched.sum())

    rng = np.random.default_rng(SEED)
    by_market = (
        valid.group_by("market_id")
        .agg(
            pl.col("sign_agreement").alias("sa"),
            pl.col("n_matched").cast(pl.Float64).alias("nm"),
        )
        .with_columns(pl.col("sa").list.len().alias("k"))
    )
    market_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for row in by_market.iter_rows(named=True):
        sa = np.asarray(row["sa"], dtype=float)
        nm = np.asarray(row["nm"], dtype=float)
        market_arrays.append((sa, nm))
    n_clusters = len(market_arrays)

    boots_simple = np.empty(N_BOOT)
    boots_vw = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        sa_concat = np.concatenate([market_arrays[i][0] for i in idx])
        nm_concat = np.concatenate([market_arrays[i][1] for i in idx])
        boots_simple[b] = sa_concat.mean()
        if nm_concat.sum() > 0:
            boots_vw[b] = (sa_concat * nm_concat).sum() / nm_concat.sum()
        else:
            boots_vw[b] = float("nan")

    lo_s, hi_s = (
        float(np.percentile(boots_simple, 2.5)),
        float(np.percentile(boots_simple, 97.5)),
    )
    lo_v, hi_v = (
        float(np.percentile(boots_vw, 2.5)),
        float(np.percentile(boots_vw, 97.5)),
    )
    se_s = float(boots_simple.std(ddof=1))
    se_v = float(boots_vw.std(ddof=1))

    summary = (
        f"§7.1 panel-level sign-agreement CI (cluster bootstrap on market_id)\n"
        f"  valid cells: {n_cells} / 400\n"
        f"  unique markets: {n_markets}\n"
        f"  total matched buckets: {total_matched:,}\n\n"
        f"  simple panel mean: {sa_simple:.4f}  "
        f"SE={se_s:.4f}  95% CI [{lo_s:.4f}, {hi_s:.4f}]\n"
        f"  volume-weighted panel mean (by n_matched): {sa_vw:.4f}  "
        f"SE={se_v:.4f}  95% CI [{lo_v:.4f}, {hi_v:.4f}]\n\n"
        f"  bootstrap iterations: {N_BOOT:,}  seed: {SEED}\n"
    )
    print(summary, flush=True)
    OUT.write_text(summary)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
