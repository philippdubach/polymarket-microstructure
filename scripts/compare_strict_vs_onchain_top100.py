"""Plan 3d-bis T3: STRICT vs on-chain measure comparison on the
top-100 stratum, full 28-day scrape window.

Promotes the n=3 anecdote from §7.2 into a distributional view that
supports the methodological-contribution claim. Output:
artifacts/measures_compare_top100.parquet with per-(market, measure,
trade_source) rows.
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.events import parse_event
from polydata.measures.effective import effective_spread, realized_spread
from polydata.measures.impact import amihud_illiquidity, kyle_lambda
from polydata.measures.spread_est import abdi_ranaldo_spread, roll_implied_spread
from polydata.onchain.token_map import load_token_map
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow

MEASURES = [
    ("effective_spread", effective_spread),
    ("realized_spread", realized_spread),
    ("abdi_ranaldo_spread", abdi_ranaldo_spread),
    ("roll_implied_spread", roll_implied_spread),
    ("kyle_lambda", kyle_lambda),
    ("amihud_illiquidity", amihud_illiquidity),
]


def _parse_market_events(df: pl.DataFrame) -> list[StreamRecord]:
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
    t_start = datetime.fromisoformat(
        os.environ.get("CALIB_START_ISO", "2026-02-28T00:00:00+00:00"),
    )
    t_end = datetime.fromisoformat(
        os.environ.get("CALIB_END_ISO", "2026-03-28T00:00:00+00:00"),
    )
    out_path_str = os.environ.get(
        "CALIB_OUT_PARQUET", "artifacts/measures_compare_top100.parquet",
    )
    print(f"window: {t_start} -> {t_end}", flush=True)
    panel = pl.read_parquet("data/panel.parquet").filter(
        pl.col("stratum") == "top"
    )
    tm = load_token_map()
    panel = panel.join(tm, on="market_id", how="inner")
    print(f"top-stratum markets: {panel.height}", flush=True)

    agg = pl.read_parquet("data/panel_onchain_cache.parquet")
    agg_window = agg.filter(
        (pl.col("t") >= t_start) & (pl.col("t") < t_end)
    )

    shard_glob = "data/panel_orderbook_shards/*.parquet"
    rows: list[pl.DataFrame] = []
    for i, r in enumerate(panel.iter_rows(named=True), start=1):
        try:
            slice_df = (
                pl.scan_parquet(shard_glob)
                .filter(pl.col("market_id") == r["market_id"])
                .filter(
                    (pl.col("timestamp_received") >= t_start)
                    & (pl.col("timestamp_received") < t_end)
                )
                .sort("timestamp_received")
                .collect()
            )
            recs = _parse_market_events(slice_df)
            w = MeasurementWindow(
                market_id=r["market_id"], t_start=t_start, t_end=t_end,
                events=recs, sample_step=timedelta(seconds=60),
            )
            onchain = agg_window.filter(
                pl.col("token_id") == r["yes_token_id"]
            ).with_columns(pl.lit(r["market_id"]).alias("market_id")).select([
                "market_id", "t", "token_id", "price", "size", "sign",
            ])
            if i % 10 == 0 or i == 1:
                print(
                    f"[{i}/{panel.height}] {r['market_id'][:14]}… "
                    f"events={len(recs):,} trades={onchain.height:,}",
                    flush=True,
                )
            for source_name, src in [("strict", None), ("onchain", onchain)]:
                for name, fn in MEASURES:
                    df = fn(w, trades=src)
                    if df.height == 0:
                        continue
                    rows.append(df.with_columns(
                        pl.lit(name).alias("measure"),
                        pl.lit(r["market_id"]).alias("market_id"),
                        pl.lit(source_name).alias("trade_source"),
                    ))
        except Exception:
            print(f"[{i}/{panel.height}] {r['market_id'][:14]}… FAILED",
                  flush=True)
            traceback.print_exc()
            continue
        if i % 25 == 0 and rows:
            partial = pl.concat(rows, how="diagonal_relaxed")
            partial.write_parquet(out_path_str)
            print(f"  checkpoint: {partial.height:,} rows", flush=True)

    panel_m = pl.concat(rows, how="diagonal_relaxed")
    out = Path(out_path_str)
    panel_m.write_parquet(out)
    print(f"\nwrote {panel_m.height:,} rows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
