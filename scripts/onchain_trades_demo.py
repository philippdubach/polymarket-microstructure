"""Plan 3a demo: load authoritative on-chain trades for the top-3 markets
by on-chain trade count in the Feb-28 → Mar-27 scrape window.

Note: the original Plan 2b top-3 demo markets (picked from an Apr 13
archive hour) have zero on-chain trades in our scrape window, so we
select by on-chain volume within the scrape window instead. Both sets are
fully resolvable via the CLOB-REST-sourced `data/clob_token_map.parquet`
(~100% coverage of the archive).

Loads the 14 GB on-chain fills directory once via lazy scan, filters to the
union of the 6 tokens (YES+NO for each of 3 markets), then aggregates +
attaches block timestamps per market.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from polydata.onchain.join import aggregate_onchain_df
from polydata.onchain.rpc import PolygonRpcClient
from polydata.onchain.token_map import load_token_map

TOP_MARKETS = [
    "0x9352c559e9648ab4cab236087b64ca85c5b7123a4c7d9d7d4efde4a39c18056f",
    "0x0b4cc3b739e1dfe5d73274740e7308b6fb389c5af040c3a174923d928d134bee",
    "0xbb4d51e6364066d92eb6f9b8413dd7193de70966736044463b205834805a1f3b",
]


def main() -> int:
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    token_map = load_token_map()
    if token_map.height == 0:
        print("token map is empty; run scripts/pull_token_ids.py first")
        return 1

    tokens_by_market: dict[str, tuple[str, str]] = {}
    wanted_tokens: set[str] = set()
    for mid in TOP_MARKETS:
        row = token_map.filter(pl.col("market_id") == mid)
        if row.height == 0:
            print(f"{mid[:12]}… not in token map; skipping")
            continue
        r = row.row(0, named=True)
        tokens_by_market[mid] = (r["yes_token_id"], r["no_token_id"])
        wanted_tokens.update({r["yes_token_id"], r["no_token_id"]})
        print(
            f"{mid[:12]}…  yes={r['yes_token_id'][:20]}… "
            f"no={r['no_token_id'][:20]}…"
        )

    if not tokens_by_market:
        print("no markets resolvable")
        return 1

    files = sorted(Path("data/onchain_fills").glob("order_filled_*.parquet"))
    print(f"\nscanning {len(files)} onchain parquet files, filtering to "
          f"{len(wanted_tokens)} tokens…")
    raw = (
        pl.scan_parquet(files)
        .filter(
            pl.col("maker_asset_id").is_in(list(wanted_tokens))
            | pl.col("taker_asset_id").is_in(list(wanted_tokens))
        )
        .collect(engine="streaming")
    )
    print(f"raw fills for wanted tokens: {raw.height:,}")

    agg = aggregate_onchain_df(raw)
    print(f"aggregated (bucketed) trades: {agg.height:,}")

    anchor_block_raw = agg["block_number"].min()
    assert anchor_block_raw is not None
    anchor_block = int(anchor_block_raw)  # type: ignore[arg-type]
    with PolygonRpcClient(rpc_url) as client:
        anchor_ts_sec = client.block_timestamp(anchor_block)
    anchor_ts = datetime.fromtimestamp(anchor_ts_sec, tz=UTC)
    agg = agg.with_columns(
        (pl.lit(anchor_ts) + pl.duration(
            seconds=(pl.col("block_number").cast(pl.Int64) - anchor_block) * 2,
        )).alias("t")
    )
    print(f"block-ts anchor: block {anchor_block} = {anchor_ts.isoformat()}")

    all_frames: list[pl.DataFrame] = []
    for mid, (yes, no) in tokens_by_market.items():
        mkt_tokens = {yes, no}
        df = agg.filter(pl.col("token_id").is_in(list(mkt_tokens)))
        df = df.with_columns(pl.lit(mid).alias("market_id"))
        vol = float((df["price"] * df["size"]).sum())
        print(
            f"  {mid[:12]}…: {df.height:,} trades  "
            f"(+1={int((df['sign'] == 1).sum()):,} "
            f"-1={int((df['sign'] == -1).sum()):,})  "
            f"vol=${vol:,.0f}"
        )
        all_frames.append(df)

    combined = pl.concat(all_frames)
    Path("artifacts").mkdir(exist_ok=True)
    out = Path("artifacts/onchain_trades_top3_demo.parquet")
    combined.write_parquet(out)
    print(f"\nwrote {combined.height:,} rows to {out}")
    print(f"total volume (USDC): {(combined['price'] * combined['size']).sum():,.0f}")
    print(f"unique markets: {combined['market_id'].n_unique()}")
    print(f"time range: {combined['t'].min()} → {combined['t'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
