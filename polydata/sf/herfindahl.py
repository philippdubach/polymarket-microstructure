"""SF4: per-market Herfindahl (HHI) of maker-address share of on-chain
trade volume. Captures liquidity concentration: HHI close to 1 means a
single maker dominates; close to 1/n_makers means even distribution."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from polydata.onchain.token_map import load_token_map


def hhi_per_market(trades: pl.DataFrame) -> pl.DataFrame:
    """Compute volume-weighted maker HHI per market."""
    if trades.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8, "hhi": pl.Float64,
                "n_makers": pl.UInt32, "total_size": pl.Float64,
            },
        )
    per_maker = (
        trades.group_by(["market_id", "maker"])
        .agg(pl.col("size").sum().alias("maker_size"))
    )
    totals = per_maker.group_by("market_id").agg(
        pl.col("maker_size").sum().alias("total_size")
    )
    share = per_maker.join(totals, on="market_id", how="inner").with_columns(
        (pl.col("maker_size") / pl.col("total_size")).alias("share")
    )
    return (
        share.group_by("market_id")
        .agg(
            (pl.col("share") ** 2).sum().alias("hhi"),
            pl.len().alias("n_makers").cast(pl.UInt32),
            pl.col("total_size").first().alias("total_size"),
        )
        .sort("hhi", descending=True)
    )


def _load_panel_maker_trades(
    onchain_dir: Path,
    panel_tokens: pl.DataFrame,
) -> pl.DataFrame:
    """Stream-scan the on-chain shards, keep (token_id, maker, size, t)
    rows for tokens in panel. Used by run_sf4 only."""
    files = sorted(onchain_dir.glob("order_filled_*.parquet"))
    if not files:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8, "maker": pl.Utf8, "size": pl.Float64,
            },
        )
    yes_tokens = panel_tokens["yes_token_id"].drop_nulls().to_list()
    no_tokens = panel_tokens["no_token_id"].drop_nulls().to_list()
    wanted = list(set(yes_tokens) | set(no_tokens))
    raw = (
        pl.scan_parquet(files)
        .filter(
            pl.col("maker_asset_id").is_in(wanted)
            | pl.col("taker_asset_id").is_in(wanted)
        )
        .with_columns(
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.col("maker_asset_id"))
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.col("taker_asset_id"))
              .otherwise(None)
              .alias("token_id"),
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.col("maker_amount_filled").cast(pl.Float64) / 1e6)
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.col("taker_amount_filled").cast(pl.Float64) / 1e6)
              .otherwise(0.0)
              .alias("size"),
        )
        .filter(pl.col("token_id").is_not_null())
        .select(["token_id", "maker", "size"])
        .collect(engine="streaming")
    )
    yes = panel_tokens.select(
        pl.col("market_id"),
        pl.col("yes_token_id").alias("token_id"),
    ).filter(pl.col("token_id").is_not_null())
    no = panel_tokens.select(
        pl.col("market_id"),
        pl.col("no_token_id").alias("token_id"),
    ).filter(pl.col("token_id").is_not_null())
    token_to_market = pl.concat([yes, no])
    return (
        raw.join(token_to_market, on="token_id", how="inner")
        .select(["market_id", "maker", "size"])
    )


def run_sf4(
    onchain_dir: Path = Path("data/onchain_fills"),
    panel_path: Path = Path("data/panel.parquet"),
    out_parquet: Path = Path("artifacts/sf4_herfindahl.parquet"),
    out_png: Path = Path("artifacts/sf4_herfindahl.png"),
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    panel = pl.read_parquet(panel_path)
    tm = load_token_map()
    panel_tokens = panel.join(tm, on="market_id", how="inner")
    print(f"panel tokens resolved: {panel_tokens.height}/{panel.height}")

    trades = _load_panel_maker_trades(onchain_dir, panel_tokens)
    print(f"trades joined to markets: {trades.height:,}")
    summary = hhi_per_market(trades)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = summary["hhi"].to_numpy()
    ax.hist(vals, bins=40, edgecolor="black")
    ax.set_xlabel("HHI of maker-address volume share")
    ax.set_ylabel("# markets")
    ax.set_title("SF4 — liquidity concentration (maker HHI)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary
