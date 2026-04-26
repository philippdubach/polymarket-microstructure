"""T3 driver: compute six trade-based measures for every panel market,
using authoritative on-chain trades. Writes data/panel_trade_measures.parquet.

Key efficiency: pre-loads all panel-market rows from the 25 archive files
in a SINGLE lazy scan (filter on market_id IN panel.market_id), avoiding
the 15k file-scans that per-market `MarketStream` would cost. Same pattern
for on-chain fills — one streaming scan filtered to all panel tokens,
sliced per-market in memory.

Scope window: env CALIB_START_ISO / CALIB_END_ISO, default 2026-02-28 →
2026-03-28.
"""
from __future__ import annotations

import os
import re
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.events import parse_event
from polydata.measures.effective import effective_spread, realized_spread
from polydata.measures.impact import amihud_illiquidity, kyle_lambda
from polydata.measures.spread_est import abdi_ranaldo_spread, roll_implied_spread
from polydata.onchain.join import aggregate_onchain_df
from polydata.onchain.rpc import PolygonRpcClient
from polydata.onchain.token_map import load_token_map
from polydata.paths import parquet_files
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow

_PAT = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")

MEASURES = [
    ("effective_spread", effective_spread),
    ("realized_spread", realized_spread),
    ("abdi_ranaldo_spread", abdi_ranaldo_spread),
    ("roll_implied_spread", roll_implied_spread),
    ("kyle_lambda", kyle_lambda),
    ("amihud_illiquidity", amihud_illiquidity),
]


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


def _load_panel_orderbook_df(
    files: list[Path],
    market_ids: list[str],
    t_start: datetime,
    t_end: datetime,
    shard_dir: Path = Path("data/panel_orderbook_shards"),
) -> pl.DataFrame:
    """Per-file scan + filter, written to per-file shards. Concatenate
    shards at the end. Lets us resume if killed; bounds peak memory to one
    file's worth at a time."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    market_set = pl.Series(values=market_ids)
    for i, f in enumerate(files, start=1):
        shard = shard_dir / f"{f.stem}.parquet"
        if shard.exists():
            continue
        try:
            df = (
                pl.scan_parquet(f)
                .filter(pl.col("market_id").is_in(market_set))
                .filter(
                    (pl.col("timestamp_received") >= t_start)
                    & (pl.col("timestamp_received") < t_end)
                )
                .collect()
            )
        except Exception as e:
            print(f"  [{i}/{len(files)}] {f.name}: ERROR {e}", flush=True)
            continue
        df.write_parquet(shard)
        if i % 24 == 0 or i == len(files):
            print(
                f"  [{i}/{len(files)}] {f.name}: {df.height:,} rows "
                f"(shard written)",
                flush=True,
            )
    # Don't concatenate — return shard_dir; per-market reads happen via
    # lazy scan with predicate pushdown to avoid materializing the full
    # frame in memory.
    print(
        f"  shards ready in {shard_dir} ({len(list(shard_dir.glob('*.parquet')))} files)",
        flush=True,
    )
    # Return marker DataFrame; main loop reads shards lazily per-market.
    return pl.DataFrame(
        schema={
            "timestamp_received": pl.Datetime("ms", "UTC"),
            "timestamp_created_at": pl.Datetime("ms", "UTC"),
            "market_id": pl.Utf8,
            "update_type": pl.Utf8,
            "data": pl.Utf8,
        },
    )


def _parse_market_events(df_for_market: pl.DataFrame) -> list[StreamRecord]:
    """Parse JSON events for a single market slice; drop non-YES."""
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


def _load_panel_onchain(
    yes_tokens: list[str], rpc_url: str,
) -> pl.DataFrame:
    """Single streaming-scan load for all panel YES tokens."""
    files = sorted(Path("data/onchain_fills").glob("order_filled_*.parquet"))
    raw = (
        pl.scan_parquet(files)
        .filter(
            pl.col("maker_asset_id").is_in(yes_tokens)
            | pl.col("taker_asset_id").is_in(yes_tokens)
        )
        .collect(engine="streaming")
    )
    agg = aggregate_onchain_df(raw)
    if agg.height == 0:
        return agg.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("t"))
    anchor_raw = agg["block_number"].min()
    assert anchor_raw is not None
    anchor = int(anchor_raw)  # type: ignore[arg-type]
    with PolygonRpcClient(rpc_url) as client:
        anchor_ts_sec = client.block_timestamp(anchor)
    anchor_ts = datetime.fromtimestamp(anchor_ts_sec, tz=UTC)
    return agg.with_columns(
        (pl.lit(anchor_ts) + pl.duration(
            seconds=(pl.col("block_number").cast(pl.Int64) - anchor) * 2,
        )).alias("t")
    )


def main() -> int:
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    t_start = datetime.fromisoformat(
        os.environ.get("CALIB_START_ISO", "2026-02-28T00:00:00+00:00"),
    )
    t_end = datetime.fromisoformat(
        os.environ.get("CALIB_END_ISO", "2026-03-28T00:00:00+00:00"),
    )
    print(f"window: {t_start} -> {t_end}")

    panel = pl.read_parquet("data/panel.parquet")
    tm = load_token_map()
    joined = panel.join(tm, on="market_id", how="inner")
    print(
        f"panel markets resolved to tokens: {joined.height}/{panel.height}"
    )

    yes_by_market: dict[str, str] = {
        r["market_id"]: r["yes_token_id"]
        for r in joined.iter_rows(named=True)
    }
    market_ids = list(yes_by_market.keys())
    yes_tokens = list(yes_by_market.values())

    files = _files_in_window(t_start, t_end)
    print(f"archive files in window: {len(files)}", flush=True)
    shard_dir = Path("data/panel_orderbook_shards")
    if shard_dir.exists() and len(list(shard_dir.glob("*.parquet"))) >= len(files):
        print(
            f"using existing shard cache: {shard_dir} "
            f"({len(list(shard_dir.glob('*.parquet')))} shards)",
            flush=True,
        )
    else:
        print(
            f"building panel orderbook shards across {len(files)} files…",
            flush=True,
        )
        _load_panel_orderbook_df(files, market_ids, t_start, t_end, shard_dir)

    onchain_cache = Path("data/panel_onchain_cache.parquet")
    if onchain_cache.exists():
        print(f"using cached on-chain agg: {onchain_cache}", flush=True)
        agg = pl.read_parquet(onchain_cache)
    else:
        print(
            f"loading on-chain fills for {len(yes_tokens)} panel tokens…",
            flush=True,
        )
        agg = _load_panel_onchain(yes_tokens, rpc_url)
        agg.write_parquet(onchain_cache)
    agg_window = agg.filter(
        (pl.col("t") >= t_start) & (pl.col("t") < t_end)
    )
    print(f"  on-chain in window: {agg_window.height:,}", flush=True)

    shard_glob = str(shard_dir / "*.parquet")
    rows: list[pl.DataFrame] = []
    for i, (mid, yes_tok) in enumerate(yes_by_market.items(), start=1):
        try:
            market_slice = (
                pl.scan_parquet(shard_glob)
                .filter(pl.col("market_id") == mid)
                .sort("timestamp_received")
                .collect()
            )
            recs = _parse_market_events(market_slice)
            w = MeasurementWindow(
                market_id=mid, t_start=t_start, t_end=t_end,
                events=recs, sample_step=timedelta(seconds=60),
            )
            onchain = agg_window.filter(
                pl.col("token_id") == yes_tok
            ).with_columns(pl.lit(mid).alias("market_id")).select([
                "market_id", "t", "token_id", "price", "size", "sign",
            ])
            if i % 25 == 0 or i == 1:
                print(
                    f"[{i}/{len(yes_by_market)}] {mid[:14]}… "
                    f"events={len(recs):,} trades={onchain.height:,}"
                )
            for name, fn in MEASURES:
                df = fn(w, trades=onchain)
                if df.height == 0:
                    continue
                rows.append(df.with_columns(
                    pl.lit(name).alias("measure"),
                    pl.lit(mid).alias("market_id"),
                ))
        except Exception:
            print(f"[{i}/{len(yes_by_market)}] {mid[:14]}… FAILED")
            traceback.print_exc()
            continue

        # Checkpoint every 50 markets so progress isn't lost on crash.
        if i % 50 == 0 and rows:
            partial = pl.concat(rows, how="diagonal_relaxed")
            partial.write_parquet("data/panel_trade_measures.parquet")
            print(f"  checkpoint: {partial.height:,} rows written")

    if not rows:
        print("no measure rows produced")
        return 1
    panel_m = pl.concat(rows, how="diagonal_relaxed")
    out = Path("data/panel_trade_measures.parquet")
    panel_m.write_parquet(out)
    print(f"\nwrote {panel_m.height:,} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
