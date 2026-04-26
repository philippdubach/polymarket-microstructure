import os
from pathlib import Path

import pytest


def test_load_onchain_trades_end_to_end_on_scraped_data():
    onchain_dir = Path("data/onchain_fills")
    slices = list(onchain_dir.glob("order_filled_*.parquet"))
    if not slices:
        pytest.skip("no scraped slices; run scripts/scrape_onchain_fills.py first")

    # Pick a known-active token id from the scraped data for validation.
    # We read one slice, take the most frequent (non-USDC) asset id.
    import polars as pl

    from polydata.onchain.trades import load_onchain_trades

    df = pl.read_parquet(slices[0])
    # Asset ids across maker side, excluding 0 (USDC)
    non_usdc = (
        df.select(pl.col("maker_asset_id"))
        .filter(pl.col("maker_asset_id") != "0")
        .group_by("maker_asset_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    if non_usdc.height == 0:
        pytest.skip("first slice has no non-USDC asset ids")
    top_token = non_usdc.row(0, named=True)["maker_asset_id"]

    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")

    trades = load_onchain_trades(
        market_id="integration_test",
        yes_token_id=top_token,
        no_token_id=None,
        onchain_dir=onchain_dir,
        rpc_url=rpc_url,
    )
    # Hard invariants
    assert trades.height > 0, f"no trades decoded for token {top_token}"
    assert set(trades["sign"].unique().to_list()).issubset({-1, 1})
    # All prices in [0,1]
    assert trades["price"].min() > 0
    assert trades["price"].max() < 1
    # Block numbers monotonically increasing after sort
    sorted_df = trades.sort("block_number")
    bns = sorted_df["block_number"].to_list()
    assert all(bns[i] <= bns[i + 1] for i in range(len(bns) - 1))
