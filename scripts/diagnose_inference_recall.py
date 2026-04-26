"""Diagnostic: measure upper-bound recall of orderbook-only trade inference
against on-chain OrderFilled ground truth.

Uses LOOSE inference (every resting-size decrement = inferred trade, no
co-timing requirement, no top-of-book restriction) to produce the maximum
number of inference candidates, then block-level-matches against the same
onchain slice the calibration uses.

Three outcomes tell us different things about STRICT's viability:

  1. LOOSE recall > 80% → our feed has the trades; STRICT is over-strict;
     relax STRICT's co-timing / top-of-book requirements.
  2. LOOSE recall 10-80% → our feed captures most trades but misses some;
     need to investigate which onchain trades lack feed signatures.
  3. LOOSE recall < 10% → our feed fundamentally lacks the events that
     correspond to on-chain trades. Polymarket's public WebSocket feed
     doesn't expose enough information to reconstruct trades. This is a
     real paper finding that invalidates inference-based measurement of
     trade-derived microstructure on Polymarket.

Env vars: POLYGON_RPC_URL, CALIB_START_ISO, CALIB_END_ISO (can be narrower
than the scrape window — this diagnostic uses the whole specified window
as one pass, no split).
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.measures.trades import aggregate_trades_to_blocks, infer_trades_loose
from polydata.onchain.join import aggregate_onchain_df, block_level_match, compute_metrics
from polydata.onchain.rpc import PolygonRpcClient
from polydata.paths import parquet_files
from polydata.stream import MarketStream
from polydata.window import MeasurementWindow

_PAT = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")
BS_SECONDS = 5  # bucket size for matching; larger = tolerates more timing skew


def _files_in_window(t_start: datetime, t_end: datetime) -> list[Path]:
    out: list[Path] = []
    for f in parquet_files():
        m = _PAT.match(f.name)
        if not m:
            continue
        file_hour = datetime.fromisoformat(m.group(1)).replace(hour=int(m.group(2)), tzinfo=UTC)
        if t_start - timedelta(hours=1) <= file_hour < t_end:
            out.append(f)
    return out


def _pick_slice_markets(t_start: datetime, t_end: datetime, n: int = 3) -> list[str]:
    files = _files_in_window(t_start, t_end)
    return (
        pl.scan_parquet(files)
        .group_by("market_id").agg(pl.len().alias("n"))
        .sort("n", descending=True).limit(n)
        .collect(engine="streaming")["market_id"].to_list()
    )


def _attach_block_ts_linear(df: pl.DataFrame, rpc_url: str) -> pl.DataFrame:
    anchor = int(df["block_number"].min())
    with PolygonRpcClient(rpc_url) as client:
        anchor_ts_sec = client.block_timestamp(anchor)
    anchor_ts = datetime.fromtimestamp(anchor_ts_sec, tz=UTC)
    return df.with_columns(
        (pl.lit(anchor_ts) + pl.duration(
            seconds=(pl.col("block_number").cast(pl.Int64) - anchor) * 2
        )).alias("block_ts")
    )


def main() -> int:
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    start_iso = os.environ.get("CALIB_START_ISO", "2026-02-28T00:00:00+00:00")
    end_iso = os.environ.get("CALIB_END_ISO", "2026-03-02T00:00:00+00:00")
    t_start = datetime.fromisoformat(start_iso)
    t_end = datetime.fromisoformat(end_iso)
    print(f"diagnostic window: {t_start} -> {t_end}  (bs={BS_SECONDS}s)")

    files = sorted(Path("data/onchain_fills").glob("order_filled_*.parquet"))
    if not files:
        raise RuntimeError("no onchain parquets; run scraper first")
    fills_df = pl.concat([pl.read_parquet(f) for f in files])
    print(f"fills loaded: {fills_df.height:,}")
    fills_df = _attach_block_ts_linear(fills_df, rpc_url)
    oc_agg = aggregate_onchain_df(fills_df.drop("block_ts"))
    ts_map = fills_df.select(["block_number", "block_ts"]).unique(subset=["block_number"])
    oc_agg = oc_agg.join(ts_map, on="block_number", how="left")
    print(f"oc_agg rows: {oc_agg.height:,}")
    del fills_df

    oc_window = oc_agg.filter(
        (pl.col("block_ts") >= t_start) & (pl.col("block_ts") < t_end)
    )
    oc_window_all = oc_window.with_columns(
        (pl.col("block_ts").dt.timestamp("ms") // (BS_SECONDS * 1000))
        .cast(pl.UInt64).alias("block_number")
    ).select(["block_number", "tx_hash", "token_id", "price", "size", "sign"])
    print(f"onchain in window (all markets, both sides): {oc_window.height:,}")

    market_ids = _pick_slice_markets(t_start, t_end, n=3)
    print(f"top-3 markets: {market_ids}")

    # Filter onchain to the token_ids of these 3 markets' YES tokens.
    # We don't know the YES token_id without the Gamma metadata; use the
    # full set and accept the FN inflation (will note in output).
    archive_files = _files_in_window(t_start, t_end)
    print(f"  archive files in window: {len(archive_files)}")

    loose_frames: list[pl.DataFrame] = []
    for i, mid in enumerate(market_ids, start=1):
        recs = [
            r for r in MarketStream(market_id=mid, files=archive_files, side="YES")
            if t_start <= r.ts_received < t_end
        ]
        print(f"  [{i}/3] {mid[:16]}… {len(recs):,} events on YES side")
        w = MeasurementWindow(
            market_id=mid, t_start=t_start, t_end=t_end, events=recs
        )
        df = infer_trades_loose(w)
        print(f"         loose inferred: {df.height:,} raw (pre-aggregation)")
        if df.height > 0:
            agg = aggregate_trades_to_blocks(df, block_seconds=BS_SECONDS)
            print(f"         after bs={BS_SECONDS}s aggregation: {agg.height:,}")
            loose_frames.append(agg)

    if not loose_frames:
        print("NO LOOSE INFERRED TRADES. Something very wrong.")
        return 2

    inferred = pl.concat(loose_frames).with_columns(
        (pl.col("t").dt.timestamp("ms") // (BS_SECONDS * 1000))
        .cast(pl.UInt64).alias("block_number")
    ).select(["market_id", "block_number", "token_id", "price", "size", "sign"])
    print("\n=== LOOSE INFERRED ===")
    print(f"  total buckets: {inferred.height:,}")

    # Filter onchain to the YES tokens observed in inferred
    inferred_tokens = set(inferred["token_id"].unique().to_list())
    oc_yes_tokens_only = oc_window_all.filter(
        pl.col("token_id").is_in(list(inferred_tokens))
    )
    print(f"  onchain restricted to YES-only token_ids: {oc_yes_tokens_only.height:,}")

    result = block_level_match(inferred, oc_yes_tokens_only)
    m = compute_metrics(result)
    print("\n=== MATCH (LOOSE vs ONCHAIN, YES-only tokens) ===")
    print(f"  n_inferred buckets: {m['n_inferred']:,}")
    print(f"  n_onchain  buckets: {m['n_onchain']:,}")
    print(f"  TP: {m['n_tp']:,}")
    print(f"  FP: {m['n_fp']:,}")
    print(f"  FN: {m['n_fn']:,}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  sign_agreement: {m['sign_agreement']:.4f}")
    print(f"  size_mae:  {m['size_mae']:.2f}")

    Path("artifacts").mkdir(exist_ok=True)
    with Path("artifacts/diagnostic_report.md").open("w") as fh:
        fh.write("# Inference-vs-Onchain Diagnostic\n\n")
        fh.write(f"Window: {t_start} → {t_end}  (bs={BS_SECONDS}s)\n\n")
        fh.write(f"Markets: {market_ids}\n\n")
        fh.write("## LOOSE inference vs Onchain (YES-only tokens)\n\n")
        fh.write(f"- inferred buckets: {m['n_inferred']:,}\n")
        fh.write(f"- onchain buckets:  {m['n_onchain']:,}\n")
        fh.write(f"- TP: {m['n_tp']:,}\n- FP: {m['n_fp']:,}\n- FN: {m['n_fn']:,}\n")
        fh.write(f"- **precision: {m['precision']:.4f}**\n")
        fh.write(f"- **recall:    {m['recall']:.4f}**\n")
        fh.write(f"- F1: {m['f1']:.4f}\n")
        fh.write(f"- sign_agreement: {m['sign_agreement']:.4f}\n")
        fh.write(f"- size_mae: {m['size_mae']:.2f}\n\n")
        fh.write("## Interpretation\n\n")
        if m["recall"] > 0.80:
            fh.write(
                "LOOSE recall > 80% → orderbook feed has the info; "
                "STRICT is over-strict.\n"
            )
        elif m["recall"] > 0.10:
            fh.write(
                "LOOSE recall 10-80% → feed captures most but not all "
                "trades; investigate misses.\n"
            )
        else:
            fh.write(
                "LOOSE recall < 10% → feed fundamentally lacks events "
                "mapping to onchain trades.\n"
            )
    print("wrote artifacts/diagnostic_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
