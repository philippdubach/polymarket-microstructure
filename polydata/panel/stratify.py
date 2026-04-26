"""Plan 3c panel stratification.

Top-100 by on-chain volume + random-500 from long tail. Selection rule
pre-registered in `docs/preregistration_plan3c.md`.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from polydata.onchain.join import SCALE
from polydata.onchain.token_map import load_token_map

RANDOM_SEED = 20260424
MIN_TRADES_FOR_RANDOM = 100


def compute_market_stats(
    onchain_dir: Path = Path("data/onchain_fills"),
) -> pl.DataFrame:
    """For every market in `clob_token_map.parquet`, count on-chain trades
    and sum USDC volume across YES+NO tokens in the scrape directory.

    Returns columns: (market_id, n_trades, volume_usd).
    """
    files = sorted(onchain_dir.glob("order_filled_*.parquet"))
    if not files:
        return pl.DataFrame(
            schema={"market_id": pl.Utf8, "n_trades": pl.UInt32,
                    "volume_usd": pl.Float64},
        )
    s = float(SCALE)
    # Lazy decode + per-token aggregation. The streaming scan pushes the
    # filter/agg through 14 GB of raw parquets without materializing a
    # single full copy in memory.
    per_token = (
        pl.scan_parquet(files)
        .with_columns([
            pl.col("maker_amount_filled").cast(pl.Float64),
            pl.col("taker_amount_filled").cast(pl.Float64),
        ])
        .with_columns([
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.col("maker_asset_id"))
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.col("taker_asset_id"))
              .otherwise(None)
              .alias("token_id"),
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.col("maker_amount_filled") / s)
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.col("taker_amount_filled") / s)
              .otherwise(0.0)
              .alias("size"),
            pl.when(pl.col("taker_asset_id") == "0")
              .then((pl.col("taker_amount_filled") / s)
                    / (pl.col("maker_amount_filled") / s))
              .when(pl.col("maker_asset_id") == "0")
              .then((pl.col("maker_amount_filled") / s)
                    / (pl.col("taker_amount_filled") / s))
              .otherwise(None)
              .alias("price"),
        ])
        .filter(pl.col("token_id").is_not_null())
        .with_columns((pl.col("price") * pl.col("size")).alias("usd"))
        .group_by("token_id")
        .agg(
            pl.len().alias("n_trades").cast(pl.UInt32),
            pl.col("usd").sum().alias("volume_usd"),
        )
        .collect(engine="streaming")
    )
    tm = load_token_map()
    yes = tm.select(
        pl.col("market_id"),
        pl.col("yes_token_id").alias("token_id"),
    )
    no = tm.select(
        pl.col("market_id"),
        pl.col("no_token_id").alias("token_id"),
    )
    token_to_market = pl.concat([yes, no]).filter(
        pl.col("token_id").is_not_null()
    )
    return (
        per_token.join(token_to_market, on="token_id", how="inner")
        .group_by("market_id")
        .agg(
            pl.col("n_trades").sum().cast(pl.UInt32).alias("n_trades"),
            pl.col("volume_usd").sum().alias("volume_usd"),
        )
        .sort("volume_usd", descending=True)
    )


def select_top_n(stats: pl.DataFrame, n: int = 100) -> pl.DataFrame:
    """Top-n by volume_usd desc; tiebreak by n_trades desc, market_id asc."""
    return (
        stats.sort(
            ["volume_usd", "n_trades", "market_id"],
            descending=[True, True, False],
        )
        .head(n)
        .with_columns(
            pl.lit("top").alias("stratum"),
            pl.int_range(1, pl.len() + 1).alias("rank").cast(pl.UInt32),
        )
    )


def select_random(
    stats: pl.DataFrame,
    top: pl.DataFrame,
    n: int = 500,
    min_trades: int = MIN_TRADES_FOR_RANDOM,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """Uniform sample of size n from markets with >= min_trades that are
    not already in `top`. Returns fewer rows if universe is smaller than n."""
    universe = (
        stats.filter(pl.col("n_trades") >= min_trades)
        .join(top.select("market_id"), on="market_id", how="anti")
    )
    if universe.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8, "n_trades": pl.UInt32,
                "volume_usd": pl.Float64, "stratum": pl.Utf8,
                "rank": pl.UInt32,
            },
        )
    pick = min(n, universe.height)
    sampled = universe.sample(n=pick, seed=seed, with_replacement=False)
    return sampled.with_columns(
        pl.lit("random").alias("stratum"),
        pl.int_range(1, pl.len() + 1).alias("rank").cast(pl.UInt32),
    )


def build_panel(
    onchain_dir: Path = Path("data/onchain_fills"),
    top_n: int = 100,
    random_n: int = 500,
    min_trades: int = MIN_TRADES_FOR_RANDOM,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    stats = compute_market_stats(onchain_dir=onchain_dir)
    top = select_top_n(stats, n=top_n)
    rand = select_random(
        stats, top, n=random_n, min_trades=min_trades, seed=seed,
    )
    return pl.concat([top, rand]).select([
        "market_id", "stratum", "rank", "n_trades", "volume_usd",
    ])
