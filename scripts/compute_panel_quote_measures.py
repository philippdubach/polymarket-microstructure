"""T4 driver: compute per-market quote-measure summaries for every panel
market. Reuses the shard cache from T3 (data/panel_orderbook_shards/),
lazy-scans per market, computes Plan 2a quote measures, and writes
per-market summary rows to data/panel_quote/*.parquet.

Per-market summaries (not time series) keep panel sizes manageable:
- spread.parquet: median spread bps + mean mid per market
- depth.parquet: mean bid/ask qty at levels 1/5/10 per market
- intensity.parquet: events/sec mean per market
- latency.parquet: percentiles per market (already 1 row/market)
- clock.parquet: block alignment share per market
"""
from __future__ import annotations

import os
import re
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.events import parse_event
from polydata.measures.clock import block_alignment
from polydata.measures.depth import mean_depth_by_level
from polydata.measures.intensity import quote_update_intensity
from polydata.measures.latency import latency_distribution
from polydata.measures.spread import quoted_spread_series
from polydata.paths import parquet_files
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow

_PAT = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")


def _files_in_window(t_start: datetime, t_end: datetime) -> list[Path]:
    out: list[Path] = []
    for f in parquet_files():
        m = _PAT.match(f.name)
        if not m:
            continue
        fh = datetime.fromisoformat(m.group(1)).replace(
            hour=int(m.group(2)), tzinfo=UTC,
        )
        if t_start - timedelta(hours=1) <= fh < t_end:
            out.append(f)
    return out


def _parse_market_events(df_for_market: pl.DataFrame) -> list[StreamRecord]:
    out: list[StreamRecord] = []
    for row in df_for_market.iter_rows(named=True):
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


def _spread_summary(window: MeasurementWindow) -> pl.DataFrame:
    """Per-market median quoted spread (bps) + mean mid + n_samples."""
    series = quoted_spread_series(window)
    if series.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8, "n_samples": pl.UInt32,
                "mean_mid": pl.Float64, "median_spread_bps": pl.Float64,
                "p25_spread_bps": pl.Float64, "p75_spread_bps": pl.Float64,
            },
        )
    valid = series.filter(
        pl.col("mid").is_not_null() & pl.col("half_spread").is_not_null()
    ).with_columns(
        (pl.col("half_spread") * 2 / pl.col("mid") * 10_000).alias("spread_bps"),
    )
    if valid.height == 0:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_samples": 0,
            "mean_mid": None, "median_spread_bps": None,
            "p25_spread_bps": None, "p75_spread_bps": None,
        }], schema={
            "market_id": pl.Utf8, "n_samples": pl.UInt32,
            "mean_mid": pl.Float64, "median_spread_bps": pl.Float64,
            "p25_spread_bps": pl.Float64, "p75_spread_bps": pl.Float64,
        })
    return pl.DataFrame([{
        "market_id": window.market_id,
        "n_samples": int(valid.height),
        "mean_mid": float(valid["mid"].mean() or 0),
        "median_spread_bps": float(valid["spread_bps"].median() or 0),
        "p25_spread_bps": float(valid["spread_bps"].quantile(0.25) or 0),
        "p75_spread_bps": float(valid["spread_bps"].quantile(0.75) or 0),
    }], schema={
        "market_id": pl.Utf8, "n_samples": pl.UInt32,
        "mean_mid": pl.Float64, "median_spread_bps": pl.Float64,
        "p25_spread_bps": pl.Float64, "p75_spread_bps": pl.Float64,
    })


def _intensity_summary(window: MeasurementWindow) -> pl.DataFrame:
    series = quote_update_intensity(window, bucket=timedelta(seconds=60))
    if series.height == 0:
        return pl.DataFrame([{
            "market_id": window.market_id,
            "n_buckets": 0, "mean_updates_per_min": None,
            "p99_updates_per_min": None,
        }], schema={
            "market_id": pl.Utf8, "n_buckets": pl.UInt32,
            "mean_updates_per_min": pl.Float64,
            "p99_updates_per_min": pl.Float64,
        })
    return pl.DataFrame([{
        "market_id": window.market_id,
        "n_buckets": int(series.height),
        "mean_updates_per_min": float(series["n_updates"].mean() or 0),
        "p99_updates_per_min": float(series["n_updates"].quantile(0.99) or 0),
    }], schema={
        "market_id": pl.Utf8, "n_buckets": pl.UInt32,
        "mean_updates_per_min": pl.Float64,
        "p99_updates_per_min": pl.Float64,
    })


def main() -> int:
    t_start = datetime.fromisoformat(
        os.environ.get("CALIB_START_ISO", "2026-02-28T00:00:00+00:00"),
    )
    t_end = datetime.fromisoformat(
        os.environ.get("CALIB_END_ISO", "2026-03-28T00:00:00+00:00"),
    )
    print(f"window: {t_start} -> {t_end}", flush=True)

    panel = pl.read_parquet("data/panel.parquet")
    market_ids = panel["market_id"].to_list()
    files = _files_in_window(t_start, t_end)
    print(
        f"archive files: {len(files)}  panel markets: {panel.height}",
        flush=True,
    )

    shard_dir = Path("data/panel_orderbook_shards")
    shard_glob = str(shard_dir / "*.parquet")
    if not shard_dir.exists() or len(list(shard_dir.glob("*.parquet"))) < len(files):
        print("ERROR: orderbook shard cache missing; run T3 first", flush=True)
        return 1

    out_dir = Path("data/panel_quote")
    out_dir.mkdir(parents=True, exist_ok=True)

    spread_rows: list[pl.DataFrame] = []
    depth_rows: list[pl.DataFrame] = []
    intensity_rows: list[pl.DataFrame] = []
    latency_rows: list[pl.DataFrame] = []
    clock_rows: list[pl.DataFrame] = []

    for i, mid in enumerate(market_ids, start=1):
        try:
            slice_df = (
                pl.scan_parquet(shard_glob)
                .filter(pl.col("market_id") == mid)
                .sort("timestamp_received")
                .collect()
            )
            recs = _parse_market_events(slice_df)
            w = MeasurementWindow(
                market_id=mid, t_start=t_start, t_end=t_end,
                events=recs, sample_step=timedelta(seconds=60),
            )
            if i % 25 == 0 or i == 1:
                print(
                    f"[{i}/{len(market_ids)}] {mid[:14]}… events={len(recs):,}",
                    flush=True,
                )

            spread_rows.append(_spread_summary(w))
            depth_rows.append(
                mean_depth_by_level(w).with_columns(
                    pl.lit(mid).alias("market_id"),
                )
            )
            intensity_rows.append(_intensity_summary(w))
            latency_rows.append(latency_distribution(w))
            clock_rows.append(block_alignment(w))
        except Exception:
            print(f"[{i}/{len(market_ids)}] {mid[:14]}… FAILED", flush=True)
            traceback.print_exc()
            continue

        if i % 50 == 0:
            for name, frames in [
                ("spread", spread_rows), ("depth", depth_rows),
                ("intensity", intensity_rows), ("latency", latency_rows),
                ("clock", clock_rows),
            ]:
                if frames:
                    pl.concat(frames, how="diagonal_relaxed").write_parquet(
                        out_dir / f"{name}.parquet"
                    )
            print(f"  checkpoint: {i} markets", flush=True)

    for name, frames in [
        ("spread", spread_rows), ("depth", depth_rows),
        ("intensity", intensity_rows), ("latency", latency_rows),
        ("clock", clock_rows),
    ]:
        if frames:
            df = pl.concat(frames, how="diagonal_relaxed")
            df.write_parquet(out_dir / f"{name}.parquet")
            print(f"wrote {df.height:,} rows to {out_dir / f'{name}.parquet'}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
