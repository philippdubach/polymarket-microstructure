"""End-to-end calibration (28-day slice, 60-cell 3D grid, precision-constrained).

Prereq: scripts/scrape_onchain_fills.py has been run, populating
data/onchain_fills/. This script:

  1. Loads all slice parquets + enriches with block timestamps.
  2. Decodes + block-tx-aggregates on-chain fills.
  3. Splits: [start, start + 21d) = calibration; [start+21d, end) = validation.
  4. For each (co_ms, bs, ntop) cell, runs infer_trades_strict on top markets
     of the CALIBRATION window and block-level-matches vs on-chain. 60 cells.
  5. Selects precision-constrained winner (>=95% precision, max recall).
  6. Re-runs at winning cell on the VALIDATION window; reports precision
     delta (must stay >= 0.90 per protocol).
  7. Writes artifacts/calibration_grid.parquet, artifacts/calibration_report.md.

Env vars: POLYGON_RPC_URL, CALIB_START_ISO, CALIB_END_ISO,
    CALIB_SPLIT_ISO (optional; default is `start + 21d`; overrides when
    the scraped window is shorter than 28 days).
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.measures.trades import infer_trades_strict
from polydata.onchain.calibrate import run_precision_constrained_sweep
from polydata.onchain.join import (
    aggregate_onchain_df,
    block_level_match,
    compute_metrics,
)
from polydata.onchain.rpc import PolygonRpcClient
from polydata.paths import parquet_files
from polydata.stream import MarketStream
from polydata.window import MeasurementWindow

_PAT = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")


def _load_onchain_fills_df(out_dir: Path) -> pl.DataFrame:
    """Load all slice parquets as a single Polars DataFrame (no Python dicts)."""
    files = sorted(out_dir.glob("order_filled_*.parquet"))
    if not files:
        raise RuntimeError(f"no onchain parquets in {out_dir}; run scrape first")
    return pl.concat([pl.read_parquet(f) for f in files])


def _attach_block_ts_linear(df: pl.DataFrame, rpc_url: str) -> pl.DataFrame:
    """Vectorized linear block_ts attachment. One RPC call for the min-block
    anchor, then block_ts = anchor_ts + (block - anchor) * 2s."""
    anchor = int(df["block_number"].min())
    with PolygonRpcClient(rpc_url) as client:
        anchor_ts_sec = client.block_timestamp(anchor)
    anchor_ts = datetime.fromtimestamp(anchor_ts_sec, tz=UTC)
    # anchor_ts + seconds delta; delta in i64 ms for duration
    return df.with_columns(
        (pl.lit(anchor_ts) + pl.duration(
            seconds=(pl.col("block_number").cast(pl.Int64) - anchor) * 2
        )).alias("block_ts")
    )


def _files_in_window(t_start: datetime, t_end: datetime) -> list[Path]:
    out: list[Path] = []
    for f in parquet_files():
        m = _PAT.match(f.name)
        if not m:
            continue
        file_hour = datetime.fromisoformat(m.group(1)).replace(
            hour=int(m.group(2)), tzinfo=UTC
        )
        if t_start - timedelta(hours=1) <= file_hour < t_end:
            out.append(f)
    return out


def _pick_slice_markets(
    t_start: datetime, t_end: datetime, n: int = 3
) -> list[str]:
    files = _files_in_window(t_start, t_end)
    if not files:
        return []
    return (
        pl.scan_parquet(files)
        .group_by("market_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .limit(n)
        .collect(engine="streaming")["market_id"]
        .to_list()
    )


def _load_market_events(
    market_ids: list[str],
    t_start: datetime,
    t_end: datetime,
) -> dict[str, list]:
    """One-time load: all YES-side events per market in [t_start, t_end).

    Returned dict is reused across all grid cells to avoid re-scanning the
    archive 60x. This is the dominant I/O cost of the calibration.
    """
    files = _files_in_window(t_start, t_end)
    print(f"  pre-loading events for {len(market_ids)} markets across {len(files)} files…")
    cache: dict[str, list] = {}
    for i, mid in enumerate(market_ids, start=1):
        recs = [
            r
            for r in MarketStream(market_id=mid, files=files, side="YES")
            if t_start <= r.ts_received < t_end
        ]
        cache[mid] = recs
        print(f"  [{i}/{len(market_ids)}] {mid[:16]}… {len(recs):,} events")
    return cache


def _inferred_from_cache(
    events_by_market: dict[str, list],
    t_start: datetime,
    t_end: datetime,
    co_ms: int,
    bs: int,
    ntop: int,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for mid, recs in events_by_market.items():
        w = MeasurementWindow(
            market_id=mid, t_start=t_start, t_end=t_end, events=recs
        )
        df = infer_trades_strict(
            w, co_timing_ms=co_ms, block_seconds=bs, top_n_levels=ntop
        )
        if df.height > 0:
            frames.append(df)
    if not frames:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8,
                "block_number": pl.UInt64,
                "token_id": pl.Utf8,
                "price": pl.Float64,
                "size": pl.Float64,
                "sign": pl.Int8,
            }
        )
    stacked = pl.concat(frames)
    return stacked.with_columns(
        (pl.col("t").dt.timestamp("ms") // (bs * 1000))
        .cast(pl.UInt64)
        .alias("block_number"),
    ).select(["market_id", "block_number", "token_id", "price", "size", "sign"])


def _onchain_block_aligned(onchain_df: pl.DataFrame, bs: int) -> pl.DataFrame:
    return onchain_df.with_columns(
        (pl.col("block_ts").dt.timestamp("ms") // (bs * 1000))
        .cast(pl.UInt64)
        .alias("block_number"),
    ).select(["block_number", "tx_hash", "token_id", "price", "size", "sign"])


def main() -> int:
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    start_iso = os.environ.get("CALIB_START_ISO", "2026-02-28T00:00:00+00:00")
    end_iso = os.environ.get("CALIB_END_ISO", "2026-03-28T00:00:00+00:00")
    t_start = datetime.fromisoformat(start_iso)
    t_end = datetime.fromisoformat(end_iso)
    split_iso = os.environ.get("CALIB_SPLIT_ISO")
    split = datetime.fromisoformat(split_iso) if split_iso else t_start + timedelta(days=21)
    if not (t_start < split < t_end):
        raise RuntimeError(
            f"split {split} must lie strictly between {t_start} and {t_end}"
        )

    print(
        f"calibration: {t_start} -> {split}  /  validation: {split} -> {t_end}"
    )

    fills_df = _load_onchain_fills_df(Path("data/onchain_fills"))
    print(f"fills loaded: {fills_df.height:,}")
    fills_df = _attach_block_ts_linear(fills_df, rpc_url)
    print(f"block_ts attached; aggregating {fills_df.height:,} fills…")
    oc_agg = aggregate_onchain_df(fills_df.drop("block_ts"))
    ts_map = fills_df.select(["block_number", "block_ts"]).unique(subset=["block_number"])
    oc_agg = oc_agg.join(ts_map, on="block_number", how="left")
    print(f"oc_agg rows: {oc_agg.height:,}")
    del fills_df

    co_grid = [50, 100, 200, 500, 1000]
    bs_grid = [1, 2, 3, 5]
    nt_grid = [1, 2, 3]
    precision_floor = 0.95

    # Calibration window
    market_ids_cal = _pick_slice_markets(t_start, split, n=3)
    print(f"calibration top-3 markets: {market_ids_cal}")
    events_cal = _load_market_events(market_ids_cal, t_start, split)
    oc_cal = oc_agg.filter(
        (pl.col("block_ts") >= t_start) & (pl.col("block_ts") < split)
    )
    # Cache block-aligned onchain per bs value (independent of co/ntop)
    oc_aligned_cache: dict[int, pl.DataFrame] = {}
    # Per-cell progress
    total_cells = len(co_grid) * len(bs_grid) * len(nt_grid)
    cell_ix = {"n": 0}

    def runner(co: int, bs: int, ntop: int):
        cell_ix["n"] += 1
        if bs not in oc_aligned_cache:
            oc_aligned_cache[bs] = _onchain_block_aligned(oc_cal, bs)
        inf = _inferred_from_cache(events_cal, t_start, split, co, bs, ntop)
        result = block_level_match(inf, oc_aligned_cache[bs])
        m = compute_metrics(result)
        print(
            f"  cell {cell_ix['n']}/{total_cells} "
            f"co={co} bs={bs} ntop={ntop} "
            f"tp={m['n_tp']} fp={m['n_fp']} fn={m['n_fn']} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        )
        return result

    result = run_precision_constrained_sweep(
        runner=runner,
        co_timing_grid=co_grid,
        block_seconds_grid=bs_grid,
        top_n_levels_grid=nt_grid,
        precision_floor=precision_floor,
    )
    print(
        f"selected: co_ms={result.best_co_timing_ms}, bs={result.best_block_seconds}, "
        f"ntop={result.best_top_n_levels}, P={result.best_precision:.3f}, "
        f"R={result.best_recall:.3f}, fallback={result.fallback_used}"
    )

    # Validation window
    market_ids_val = _pick_slice_markets(split, t_end, n=3)
    print(f"validation top-3 markets: {market_ids_val}")
    events_val = _load_market_events(market_ids_val, split, t_end)
    oc_val = oc_agg.filter(
        (pl.col("block_ts") >= split) & (pl.col("block_ts") < t_end)
    )
    inf_val = _inferred_from_cache(
        events_val,
        split,
        t_end,
        result.best_co_timing_ms,
        result.best_block_seconds,
        result.best_top_n_levels,
    )
    val_metrics = compute_metrics(
        block_level_match(
            inf_val,
            _onchain_block_aligned(oc_val, result.best_block_seconds),
        )
    )
    val_ok = val_metrics["precision"] >= 0.90
    print(
        f"validation: P={val_metrics['precision']:.3f}, "
        f"R={val_metrics['recall']:.3f}, ok={val_ok}"
    )

    Path("artifacts").mkdir(exist_ok=True)
    result.grid.write_parquet("artifacts/calibration_grid.parquet")

    with Path("artifacts/calibration_report.md").open("w") as fh:
        fh.write("# Calibration Report (Plan 2c)\n\n")
        fh.write(f"Calibration window: {t_start} -> {split}\n\n")
        fh.write(f"Validation window:  {split} -> {t_end}\n\n")
        fh.write(f"Precision floor: {precision_floor}\n\n")
        fh.write("## Selected thresholds\n\n")
        fh.write(f"- co_timing_ms: **{result.best_co_timing_ms}**\n")
        fh.write(f"- block_seconds: **{result.best_block_seconds}**\n")
        fh.write(f"- top_n_levels: **{result.best_top_n_levels}**\n\n")
        fh.write("## Calibration metrics (selected cell)\n\n")
        fh.write(f"- Precision: {result.best_precision:.3f}\n")
        fh.write(f"- Recall: {result.best_recall:.3f}\n")
        fh.write(f"- F1: {result.best_f1:.3f}\n")
        fh.write(f"- Fallback used: {result.fallback_used}\n\n")
        fh.write("## Validation metrics (held-out 7d)\n\n")
        fh.write(f"- Precision: {val_metrics['precision']:.3f}\n")
        fh.write(f"- Recall: {val_metrics['recall']:.3f}\n")
        fh.write(f"- Validation precision >= 0.90: **{val_ok}**\n\n")
        fh.write("## Full grid\n\n")
        fh.write(result.grid.to_pandas().to_markdown(index=False))
        fh.write("\n")
    print(
        "wrote artifacts/calibration_grid.parquet and artifacts/calibration_report.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
