"""Plan 3b deliverable: side-by-side measure comparison for STRICT vs
on-chain authoritative trade sources.

For each top-3 market and each of the six trade-based measures, computes
the measure under both trade sources and writes a long-form parquet:
    columns: market_id, measure, trade_source, ... (measure-specific cols)

Memory: loads the 14 GB on-chain fills directory ONCE via streaming scan +
token filter, then dispatches per-market slices in memory — never loops
`load_onchain_trades` per market (3x14 GB = OOM).
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.measures.effective import effective_spread, realized_spread
from polydata.measures.impact import amihud_illiquidity, kyle_lambda
from polydata.measures.spread_est import abdi_ranaldo_spread, roll_implied_spread
from polydata.onchain.join import aggregate_onchain_df
from polydata.onchain.rpc import PolygonRpcClient
from polydata.onchain.token_map import load_token_map
from polydata.paths import parquet_files
from polydata.stream import MarketStream
from polydata.window import MeasurementWindow

# Top 3 markets by on-chain trade count in the Feb-28 -> Mar-27 scrape window
# that are ALSO in the CLOB token-map. These replace the original Plan 2b
# top-3 markets (which had zero on-chain trades in this window because they
# were picked from an Apr 13 archive hour -- outside the scrape).
TOP_MARKETS = [
    "0x9352c559e9648ab4cab236087b64ca85c5b7123a4c7d9d7d4efde4a39c18056f",
    "0x0b4cc3b739e1dfe5d73274740e7308b6fb389c5af040c3a174923d928d134bee",
    "0xbb4d51e6364066d92eb6f9b8413dd7193de70966736044463b205834805a1f3b",
]

_PAT = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")


def _files_in_window(t_start: datetime, t_end: datetime) -> list[Path]:
    out: list[Path] = []
    for f in parquet_files():
        m = _PAT.match(f.name)
        if not m:
            continue
        file_hour = datetime.fromisoformat(m.group(1)).replace(
            hour=int(m.group(2)), tzinfo=UTC,
        )
        if t_start - timedelta(hours=1) <= file_hour < t_end:
            out.append(f)
    return out


def _load_onchain_all(yes_tokens: list[str], rpc_url: str) -> pl.DataFrame:
    """Single streaming-scan load for all wanted tokens. Returns aggregated
    trades with block_ts attached; caller filters per market."""
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
    anchor_block_raw = agg["block_number"].min()
    assert anchor_block_raw is not None
    anchor_block = int(anchor_block_raw)  # type: ignore[arg-type]
    with PolygonRpcClient(rpc_url) as client:
        anchor_ts_sec = client.block_timestamp(anchor_block)
    anchor_ts = datetime.fromtimestamp(anchor_ts_sec, tz=UTC)
    return agg.with_columns(
        (pl.lit(anchor_ts) + pl.duration(
            seconds=(pl.col("block_number").cast(pl.Int64) - anchor_block) * 2,
        )).alias("t")
    )


def _measure_panel(
    window: MeasurementWindow,
    trades_strict: pl.DataFrame | None,
    trades_onchain: pl.DataFrame,
) -> pl.DataFrame:
    """Run all six measures with both trade sources, return long-form rows."""
    rows: list[pl.DataFrame] = []
    for source_name, src in [
        ("strict", trades_strict),
        ("onchain", trades_onchain),
    ]:
        for name, fn in [
            ("effective_spread", effective_spread),
            ("realized_spread", realized_spread),
            ("abdi_ranaldo_spread", abdi_ranaldo_spread),
            ("roll_implied_spread", roll_implied_spread),
            ("kyle_lambda", kyle_lambda),
            ("amihud_illiquidity", amihud_illiquidity),
        ]:
            df = fn(window, trades=src)
            if df.height == 0:
                continue
            df = df.with_columns([
                pl.lit(name).alias("measure"),
                pl.lit(source_name).alias("trade_source"),
            ])
            rows.append(df)
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows, how="diagonal_relaxed")


def main() -> int:
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    # Default: a 24h window inside the on-chain scrape (Feb 28 -> Mar 27).
    t_start = datetime.fromisoformat(
        os.environ.get("CALIB_START_ISO", "2026-03-10T00:00:00+00:00"),
    )
    t_end = datetime.fromisoformat(
        os.environ.get("CALIB_END_ISO", "2026-03-11T00:00:00+00:00"),
    )
    print(f"window: {t_start} -> {t_end}")
    token_map = load_token_map()
    if token_map.height == 0:
        print("token map empty; run scripts/pull_token_ids_clob.py first")
        return 1

    files = _files_in_window(t_start, t_end)
    print(f"archive files in window: {len(files)}")

    tokens_by_market: dict[str, str] = {}
    for mid in TOP_MARKETS:
        tm = token_map.filter(pl.col("market_id") == mid)
        if tm.height == 0:
            print(f"{mid[:16]}… no token map; skipping")
            continue
        tokens_by_market[mid] = tm.row(0, named=True)["yes_token_id"]
    if not tokens_by_market:
        print("no resolvable markets")
        return 1

    yes_tokens = list(tokens_by_market.values())
    print(f"loading on-chain fills for {len(yes_tokens)} tokens…")
    agg = _load_onchain_all(yes_tokens, rpc_url)
    agg_windowed = agg.filter(
        (pl.col("t") >= t_start) & (pl.col("t") < t_end)
    )
    print(f"  aggregated: {agg.height:,}  in-window: {agg_windowed.height:,}")

    panel_rows: list[pl.DataFrame] = []
    for mid, yes_tok in tokens_by_market.items():
        recs = [
            rec for rec in MarketStream(market_id=mid, files=files, side="YES")
            if t_start <= rec.ts_received < t_end
        ]
        w = MeasurementWindow(
            market_id=mid, t_start=t_start, t_end=t_end,
            events=recs, sample_step=timedelta(seconds=1),
        )
        onchain = agg_windowed.filter(
            pl.col("token_id") == yes_tok
        ).with_columns(pl.lit(mid).alias("market_id")).select([
            "market_id", "t", "token_id", "price", "size", "sign",
        ])
        print(f"{mid[:16]}… orderbook events: {len(recs):,}  "
              f"onchain trades: {onchain.height:,}")
        rows_df = _measure_panel(w, None, onchain)
        if rows_df.height > 0:
            panel_rows.append(rows_df)

    if not panel_rows:
        print("no panel rows produced")
        return 1
    panel = pl.concat(panel_rows, how="diagonal_relaxed")
    Path("artifacts").mkdir(exist_ok=True)
    out = Path("artifacts/measures_compare.parquet")
    panel.write_parquet(out)
    print(f"\nwrote {panel.height:,} rows to {out}")
    print(panel.group_by(["measure", "trade_source"]).agg(
        pl.len().alias("n_rows"),
    ).sort(["measure", "trade_source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
