"""Plan 3d-bis T1: extend diagnose_inference_recall to top-100 × 4
disjoint 7-day windows.

For each (window, market) cell we report:
  - LOOSE precision, recall
  - sign-agreement on matched buckets
  - bootstrap 95% CI on sign-agreement (per cell)

Panel-level aggregates (median + IQR) across all (window, market) cells
are the reviewer-required robustness statistic.
Reuses panel_orderbook_shards/ and panel_onchain_cache.parquet.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from polydata.events import parse_event
from polydata.measures.trades import aggregate_trades_to_blocks, infer_trades_loose
from polydata.onchain.token_map import load_token_map
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow

WINDOWS = [
    (datetime(2026, 2, 28, tzinfo=UTC), datetime(2026, 3, 7, tzinfo=UTC)),
    (datetime(2026, 3, 7, tzinfo=UTC),  datetime(2026, 3, 14, tzinfo=UTC)),
    (datetime(2026, 3, 14, tzinfo=UTC), datetime(2026, 3, 21, tzinfo=UTC)),
    (datetime(2026, 3, 21, tzinfo=UTC), datetime(2026, 3, 28, tzinfo=UTC)),
]
BS_SECONDS = 5
N_BOOTSTRAP = 200


def _bootstrap_ci(arr: np.ndarray, n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """95% CI on a Bernoulli mean via bootstrap."""
    if arr.size < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260424)
    samples = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, arr.size, size=arr.size)
        samples[i] = arr[idx].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _match_and_score(
    inferred: pl.DataFrame, onchain: pl.DataFrame,
) -> tuple[int, int, int, float, np.ndarray]:
    """Replicates block_level_match but returns the per-bucket
    sign-agreement boolean array so we can bootstrap CIs."""
    if inferred.height == 0 or onchain.height == 0:
        return 0, 0, 0, float("nan"), np.array([], dtype=bool)
    inf_agg = (
        inferred.group_by(["block_number", "token_id", "price"])
        .agg([
            pl.col("size").sum().alias("inferred_size"),
            pl.col("sign").first().alias("inferred_sign"),
        ])
    )
    oc_agg = (
        onchain.group_by(["block_number", "token_id", "price"])
        .agg([
            pl.col("size").sum().alias("onchain_size"),
            pl.col("sign").first().alias("onchain_sign"),
        ])
    )
    joined = inf_agg.join(
        oc_agg, on=["block_number", "token_id", "price"], how="full",
    )
    if "block_number_right" in joined.columns:
        joined = joined.drop([
            c for c in
            ["block_number_right", "token_id_right", "price_right"]
            if c in joined.columns
        ])
    tp = joined.filter(
        joined["inferred_size"].is_not_null()
        & joined["onchain_size"].is_not_null()
    )
    n_tp = int(tp.height)
    n_inferred = int(inf_agg.height)
    n_onchain = int(oc_agg.height)
    if n_tp == 0:
        return n_inferred, n_onchain, 0, float("nan"), np.array([], dtype=bool)
    agree = (tp["inferred_sign"] == tp["onchain_sign"]).to_numpy()
    sign_agreement = float(agree.mean())
    return n_inferred, n_onchain, n_tp, sign_agreement, agree


def _load_market_events(
    market_id: str, t_start: datetime, t_end: datetime,
) -> list[StreamRecord]:
    df = (
        pl.scan_parquet("data/panel_orderbook_shards/*.parquet")
        .filter(pl.col("market_id") == market_id)
        .filter(
            (pl.col("timestamp_received") >= t_start)
            & (pl.col("timestamp_received") < t_end)
        )
        .sort("timestamp_received")
        .collect()
    )
    out: list[StreamRecord] = []
    for row in df.iter_rows(named=True):
        try:
            ev = parse_event(row["data"])
        except Exception:
            continue
        if ev.side != "YES":
            continue
        out.append(StreamRecord(
            ts_received=row["timestamp_received"],
            ts_created=row["timestamp_created_at"],
            event=ev,
        ))
    return out


def main() -> int:
    panel = pl.read_parquet("data/panel.parquet").filter(
        pl.col("stratum") == "top"
    )
    tm = load_token_map()
    panel_tokens = panel.join(tm, on="market_id", how="inner").select([
        "market_id", "yes_token_id",
    ])
    print(f"top-stratum markets: {panel_tokens.height}", flush=True)

    agg = pl.read_parquet("data/panel_onchain_cache.parquet")
    rows: list[dict] = []
    for w_idx, (t_start, t_end) in enumerate(WINDOWS):
        agg_window = agg.filter(
            (pl.col("t") >= t_start) & (pl.col("t") < t_end)
        )
        print(
            f"\nwindow {w_idx + 1}/4: {t_start.date()} -> {t_end.date()}, "
            f"on-chain rows: {agg_window.height:,}", flush=True,
        )
        for i, r in enumerate(panel_tokens.iter_rows(named=True), start=1):
            mid = r["market_id"]
            yes_tok = r["yes_token_id"]
            try:
                recs = _load_market_events(mid, t_start, t_end)
                w = MeasurementWindow(
                    market_id=mid, t_start=t_start, t_end=t_end,
                    events=recs, sample_step=timedelta(seconds=60),
                )
                inf = infer_trades_loose(w)
                if inf.height == 0:
                    rows.append({
                        "market_id": mid, "window_idx": w_idx,
                        "n_inferred": 0, "n_matched": 0,
                        "sign_agreement": float("nan"),
                        "ci_lo": float("nan"), "ci_hi": float("nan"),
                    })
                    continue
                inf_buckets = aggregate_trades_to_blocks(
                    inf, block_seconds=BS_SECONDS,
                ).with_columns(
                    (pl.col("t").dt.timestamp("ms") // (BS_SECONDS * 1000))
                    .cast(pl.UInt64).alias("block_number"),
                ).select([
                    "market_id", "block_number", "token_id",
                    "price", "size", "sign",
                ])
                oc_buckets = agg_window.filter(
                    pl.col("token_id") == yes_tok
                ).with_columns(
                    (pl.col("t").dt.timestamp("ms") // (BS_SECONDS * 1000))
                    .cast(pl.UInt64).alias("block_number"),
                ).select([
                    "block_number", "tx_hash", "token_id",
                    "price", "size", "sign",
                ])
                n_inf, n_oc, n_matched, sa, agree = _match_and_score(
                    inf_buckets, oc_buckets,
                )
                ci_lo, ci_hi = _bootstrap_ci(agree)
                rows.append({
                    "market_id": mid, "window_idx": w_idx,
                    "n_inferred": int(n_inf),
                    "n_matched": int(n_matched),
                    "sign_agreement": sa,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                })
            except Exception as e:
                print(f"  [{i}/{panel_tokens.height}] {mid[:14]}… ERR {e}",
                      flush=True)
                continue
            if i % 25 == 0 or i == 1:
                last = rows[-1]
                print(
                    f"  [{i}/{panel_tokens.height}] "
                    f"{mid[:14]}… matched={last['n_matched']:,} "
                    f"sa={last['sign_agreement']:.3f}",
                    flush=True,
                )
        # Per-window checkpoint
        partial = pl.DataFrame(rows)
        partial.write_parquet("artifacts/sign_agreement_robustness.parquet")
        print(f"  window {w_idx + 1} checkpoint: "
              f"{partial.height:,} cells", flush=True)

    out = pl.DataFrame(rows)
    out_path = Path("artifacts/sign_agreement_robustness.parquet")
    out.write_parquet(out_path)
    print(f"\nwrote {out.height:,} rows to {out_path}", flush=True)
    valid = out.filter(pl.col("sign_agreement").is_not_null())
    print(f"valid cells: {valid.height} / {out.height}")
    print(
        f"panel mean sign-agreement: "
        f"{float(valid['sign_agreement'].mean() or 0):.4f}",
    )
    print(
        f"panel median:              "
        f"{float(valid['sign_agreement'].median() or 0):.4f}",
    )
    print(
        f"p25-p75: "
        f"{float(valid['sign_agreement'].quantile(0.25) or 0):.4f} - "
        f"{float(valid['sign_agreement'].quantile(0.75) or 0):.4f}",
    )
    print("\nper-window summary:")
    print(valid.group_by("window_idx").agg(
        pl.col("sign_agreement").median().alias("median_sa"),
        pl.col("sign_agreement").quantile(0.25).alias("p25"),
        pl.col("sign_agreement").quantile(0.75).alias("p75"),
        pl.len().alias("n_markets"),
    ).sort("window_idx"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
