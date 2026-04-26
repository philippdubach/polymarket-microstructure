"""Block-level tx-aggregated fuzzy match — primary calibration unit.

Per pre-registered protocol (docs/preregistration_2c.md):
- Match unit: (token_id, block_number, exact price).
- On-chain aggregation: group by `tx_hash`, volume-weighted price if
  multiple logs in same tx at different prices, sum size across legs.
- Inferred is already block-aggregated by `aggregate_trades_to_blocks`.
- A block-bucket with >=1 on-chain tx AND >=1 inferred trade = 1 TP.
- FP: inferred bucket with no on-chain tx at same (block, token, price).
- FN: on-chain bucket with no inferred trade at same (block, token, price).
- Size-MAE computed over TPs on summed size per bucket.

Price is required EXACT (after USDC/token 6-decimal normalization).
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

USDC_ASSET_ID = 0
SCALE = 10**6


def decode_fill_to_trade(fill: dict) -> dict:
    mid = int(fill["maker_asset_id"])
    tid = int(fill["taker_asset_id"])
    m_amt = int(fill["maker_amount_filled"])
    t_amt = int(fill["taker_amount_filled"])
    if tid == USDC_ASSET_ID and m_amt > 0:
        return {"block_number": fill["block_number"], "tx_hash": fill["tx_hash"],
                "token_id": str(mid),
                "price": (t_amt / SCALE) / (m_amt / SCALE),
                "size": m_amt / SCALE, "sign": 1}
    if mid == USDC_ASSET_ID and t_amt > 0:
        return {"block_number": fill["block_number"], "tx_hash": fill["tx_hash"],
                "token_id": str(tid),
                "price": (m_amt / SCALE) / (t_amt / SCALE),
                "size": t_amt / SCALE, "sign": -1}
    return {"block_number": fill["block_number"], "tx_hash": fill["tx_hash"],
            "token_id": str(mid), "price": float("nan"),
            "size": 0.0, "sign": 0}


def aggregate_onchain_by_block_tx(fills: list[dict]) -> pl.DataFrame:
    decoded = [decode_fill_to_trade(f) for f in fills]
    decoded = [d for d in decoded if d["sign"] != 0]
    if not decoded:
        return pl.DataFrame(schema={
            "block_number": pl.UInt64, "tx_hash": pl.Utf8, "token_id": pl.Utf8,
            "price": pl.Float64, "size": pl.Float64, "sign": pl.Int8,
        })
    df = pl.DataFrame(decoded, schema={
        "block_number": pl.UInt64, "tx_hash": pl.Utf8, "token_id": pl.Utf8,
        "price": pl.Float64, "size": pl.Float64, "sign": pl.Int8,
    })
    return (
        df
        .with_columns((pl.col("price") * pl.col("size")).alias("_pv"))
        .group_by(["block_number", "tx_hash", "token_id", "sign"])
        .agg([
            pl.col("size").sum().alias("size"),
            (pl.col("_pv").sum() / pl.col("size").sum()).alias("price"),
        ])
        .select(["block_number", "tx_hash", "token_id", "price", "size", "sign"])
    )


def aggregate_onchain_df(raw: pl.DataFrame) -> pl.DataFrame:
    """Polars-native variant avoiding Python-level iteration.

    Accepts the raw scraper-schema DataFrame (hex-string asset_id /
    amount_filled columns), decodes buyer/seller sign + price + size via
    vectorized expressions, filters out split-fills (sign=0), and
    aggregates by (block, tx, token, sign) with volume-weighted price.
    """
    schema = {
        "block_number": pl.UInt64, "tx_hash": pl.Utf8, "token_id": pl.Utf8,
        "price": pl.Float64, "size": pl.Float64, "sign": pl.Int8,
    }
    if raw.height == 0:
        return pl.DataFrame(schema=schema)
    s = float(SCALE)
    decoded = (
        raw
        .with_columns([
            pl.col("maker_amount_filled").cast(pl.Float64),
            pl.col("taker_amount_filled").cast(pl.Float64),
        ])
        .with_columns([
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.lit(1, dtype=pl.Int8))
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.lit(-1, dtype=pl.Int8))
              .otherwise(pl.lit(0, dtype=pl.Int8))
              .alias("sign"),
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.col("maker_asset_id"))
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.col("taker_asset_id"))
              .otherwise(pl.col("maker_asset_id"))
              .alias("token_id"),
            pl.when(pl.col("taker_asset_id") == "0")
              .then((pl.col("taker_amount_filled") / s)
                    / (pl.col("maker_amount_filled") / s))
              .when(pl.col("maker_asset_id") == "0")
              .then((pl.col("maker_amount_filled") / s)
                    / (pl.col("taker_amount_filled") / s))
              .otherwise(pl.lit(float("nan")))
              .alias("price"),
            pl.when(pl.col("taker_asset_id") == "0")
              .then(pl.col("maker_amount_filled") / s)
              .when(pl.col("maker_asset_id") == "0")
              .then(pl.col("taker_amount_filled") / s)
              .otherwise(pl.lit(0.0))
              .alias("size"),
        ])
        .filter(pl.col("sign") != 0)
        .select(["block_number", "tx_hash", "token_id", "price", "size", "sign"])
    )
    return (
        decoded
        .with_columns((pl.col("price") * pl.col("size")).alias("_pv"))
        .group_by(["block_number", "tx_hash", "token_id", "sign"])
        .agg([
            pl.col("size").sum().alias("size"),
            (pl.col("_pv").sum() / pl.col("size").sum()).alias("price"),
        ])
        .select(["block_number", "tx_hash", "token_id", "price", "size", "sign"])
    )


@dataclass(frozen=True)
class JoinResult:
    n_inferred: int
    n_onchain: int
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    size_mae: float
    sign_agreement: float


def block_level_match(
    inferred: pl.DataFrame, onchain: pl.DataFrame,
) -> JoinResult:
    if inferred.height == 0 and onchain.height == 0:
        return JoinResult(0, 0, 0, 0, 0, 0.0, 1.0)

    inf_agg = (
        inferred
        .group_by(["block_number", "token_id", "price"])
        .agg([
            pl.col("size").sum().alias("inferred_size"),
            pl.col("sign").first().alias("inferred_sign"),
            pl.len().alias("inferred_n"),
        ])
    )
    oc_agg = (
        onchain
        .group_by(["block_number", "token_id", "price"])
        .agg([
            pl.col("size").sum().alias("onchain_size"),
            pl.col("sign").first().alias("onchain_sign"),
            pl.col("tx_hash").n_unique().alias("onchain_n_tx"),
        ])
    )
    joined = inf_agg.join(
        oc_agg, on=["block_number", "token_id", "price"], how="full",
    )
    if "block_number_right" in joined.columns:
        joined = joined.with_columns([
            pl.coalesce([pl.col("block_number"),
                         pl.col("block_number_right")]).alias("block_number"),
            pl.coalesce([pl.col("token_id"), pl.col("token_id_right")]).alias("token_id"),
            pl.coalesce([pl.col("price"), pl.col("price_right")]).alias("price"),
        ]).drop(["block_number_right", "token_id_right", "price_right"])

    tp_mask = joined["inferred_size"].is_not_null() & joined["onchain_size"].is_not_null()
    fp_mask = joined["inferred_size"].is_not_null() & joined["onchain_size"].is_null()
    fn_mask = joined["inferred_size"].is_null() & joined["onchain_size"].is_not_null()

    tp = joined.filter(tp_mask)
    n_tp = int(tp.height)
    n_fp = int(joined.filter(fp_mask).height)
    n_fn = int(joined.filter(fn_mask).height)

    if n_tp > 0:
        size_mae = float((tp["inferred_size"] - tp["onchain_size"]).abs().mean())
        sign_agreement = float(
            (tp["inferred_sign"] == tp["onchain_sign"]).sum() / n_tp
        )
    else:
        size_mae = 0.0
        sign_agreement = 1.0

    return JoinResult(
        n_inferred=int(inf_agg.height), n_onchain=int(oc_agg.height),
        n_true_positive=n_tp, n_false_positive=n_fp, n_false_negative=n_fn,
        size_mae=size_mae, sign_agreement=sign_agreement,
    )


def compute_metrics(result: JoinResult) -> dict:
    tp, fp, fn = result.n_true_positive, result.n_false_positive, result.n_false_negative
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "size_mae": result.size_mae, "sign_agreement": result.sign_agreement,
        "n_inferred": result.n_inferred, "n_onchain": result.n_onchain,
        "n_tp": tp, "n_fp": fp, "n_fn": fn,
    }
