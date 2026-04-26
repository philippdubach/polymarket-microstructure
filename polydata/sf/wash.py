"""SF7: self-counterparty wash filter.

Two-tier flag (per Plan 3 design spec §5.4):
  (a) Direct self-match: maker == taker on a single trade.
  (b) Roundtrip pair: trade A with (maker_A, taker_A) and trade B with
      (taker_A, maker_A) within `window_blocks` blocks in the same
      market.

This is a pragmatic LOWER BOUND on wash activity. Sirolly et al. 2025's
network-based classifier reports ~25%; the paper §5.7 cites their figure
alongside our number.

Implementation: per-market sliding-window scan. O(n × avg-trades-in-
window) per market, ~O(panel_n_trades * 16) overall for the densest
markets in the panel.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from polydata.onchain.token_map import load_token_map

WINDOW_BLOCKS_DEFAULT = 128


def flag_self_counterparty(
    trades: pl.DataFrame, window_blocks: int = WINDOW_BLOCKS_DEFAULT,
) -> pl.DataFrame:
    """Attach `is_wash` boolean column. Trades flagged if maker==taker OR if
    a flipped (maker, taker) pair exists within window_blocks blocks in the
    same market."""
    if trades.height == 0:
        return trades.with_columns(pl.lit(False).alias("is_wash"))
    direct = (pl.col("maker") == pl.col("taker"))
    base = trades.with_columns(direct.alias("_is_wash_direct"))
    out_rows: list[dict] = []
    for _mid, sub in base.group_by("market_id"):
        ms = sub.sort("block_number").to_dicts()
        flagged = [r["_is_wash_direct"] for r in ms]
        # Sliding-window roundtrip detection.
        for i, ti in enumerate(ms):
            if flagged[i]:
                continue
            block_i = ti["block_number"]
            for j in range(i + 1, len(ms)):
                tj = ms[j]
                if tj["block_number"] - block_i > window_blocks:
                    break
                if (
                    ti["maker"] == tj["taker"]
                    and ti["taker"] == tj["maker"]
                    and ti["maker"] != ti["taker"]
                ):
                    flagged[i] = True
                    flagged[j] = True
                    break
        for r, f in zip(ms, flagged, strict=True):
            r["is_wash"] = bool(f)
            out_rows.append(r)
    return pl.DataFrame(out_rows).drop("_is_wash_direct")


def wash_share_per_market(flagged: pl.DataFrame) -> pl.DataFrame:
    if flagged.height == 0:
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8, "wash_share": pl.Float64,
                "n_trades": pl.UInt32, "n_wash": pl.UInt32,
            },
        )
    return (
        flagged.group_by("market_id")
        .agg(
            pl.col("is_wash").mean().alias("wash_share"),
            pl.len().alias("n_trades").cast(pl.UInt32),
            pl.col("is_wash").sum().alias("n_wash").cast(pl.UInt32),
        )
        .sort("wash_share", descending=True)
    )


def _load_panel_trades_with_makers(
    onchain_dir: Path, panel_tokens: pl.DataFrame,
) -> pl.DataFrame:
    files = sorted(onchain_dir.glob("order_filled_*.parquet"))
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
        .select(["block_number", "token_id", "maker", "taker", "size"])
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
    return raw.join(token_to_market, on="token_id", how="inner").select([
        "market_id", "block_number", "maker", "taker", "size",
    ])


def run_sf7(
    onchain_dir: Path = Path("data/onchain_fills"),
    panel_path: Path = Path("data/panel.parquet"),
    out_parquet: Path = Path("artifacts/sf7_wash.parquet"),
    out_png: Path = Path("artifacts/sf7_wash.png"),
    window_blocks: int = WINDOW_BLOCKS_DEFAULT,
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    panel = pl.read_parquet(panel_path)
    tm = load_token_map()
    panel_tokens = panel.join(tm, on="market_id", how="inner")
    print("loading panel maker trades…", flush=True)
    trades = _load_panel_trades_with_makers(onchain_dir, panel_tokens)
    print(f"  panel trades: {trades.height:,}", flush=True)

    print("flagging wash (maker==taker + roundtrips ≤128 blocks)…", flush=True)
    flagged = flag_self_counterparty(trades, window_blocks=window_blocks)
    summary = wash_share_per_market(flagged)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary["wash_share"].to_numpy(), bins=40, edgecolor="black")
    ax.axvline(0.25, color="red", linestyle="--",
               label="Sirolly 2025 reference (25%)")
    ax.set_xlabel("self-counterparty share per market")
    ax.set_ylabel("# markets")
    ax.set_title("SF7 — self-counterparty wash share (panel)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary
