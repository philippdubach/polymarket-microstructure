"""Authoritative on-chain trade stream for Polymarket.

Decodes scraped OrderFilled events into a canonical trade record with
authoritative aggressor sign. No inference required — `sign` is derived
directly from which side posted USDC (assetId == 0):

    takerAssetId == 0 → taker posted USDC → taker is BUYER → sign = +1
    makerAssetId == 0 → maker posted USDC → taker is SELLER → sign = -1

This replaces the orderbook-only STRICT/LOOSE inference used in Plan 2b,
which had a 57% sign-agreement problem (documented in CLAUDE.md,
2026-04-21). For any trade-direction-dependent measure (effective spread,
Kyle's λ, Amihud), use this module's output, not the inferred stream.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

ONCHAIN_TRADE_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "t": pl.Datetime(time_zone="UTC"),
    "block_number": pl.UInt64,
    "tx_hash": pl.Utf8,
    "token_id": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "sign": pl.Int8,
    "maker": pl.Utf8,
    "taker": pl.Utf8,
}


@dataclass(frozen=True)
class OnchainTrade:
    ts: datetime
    block_number: int
    tx_hash: str
    token_id: str
    price: Decimal
    size: Decimal
    sign: int  # +1 buyer-initiated, -1 seller-initiated
    maker: str
    taker: str


from polydata.onchain.join import aggregate_onchain_df  # noqa: E402
from polydata.onchain.rpc import PolygonRpcClient  # noqa: E402


def _attach_block_ts_linear(df: pl.DataFrame, rpc_url: str) -> pl.DataFrame:
    anchor = int(df["block_number"].min())
    with PolygonRpcClient(rpc_url) as client:
        anchor_ts_sec = client.block_timestamp(anchor)
    anchor_ts = datetime.fromtimestamp(anchor_ts_sec, tz=UTC)
    return df.with_columns(
        (pl.lit(anchor_ts) + pl.duration(
            seconds=(pl.col("block_number").cast(pl.Int64) - anchor) * 2,
        )).alias("t")
    )


def load_onchain_trades(
    market_id: str,
    yes_token_id: str | None,
    no_token_id: str | None,
    t_start: datetime | None = None,
    t_end: datetime | None = None,
    onchain_dir: Path = Path("data/onchain_fills"),
    rpc_url: str = "https://polygon-rpc.com",
) -> pl.DataFrame:
    """Load authoritative on-chain trades for a single market.

    Filters scraped OrderFilled parquets to the specified YES/NO token ids,
    decodes + aggregates via `aggregate_onchain_df`, attaches block
    timestamps via linear interpolation (one RPC call).

    Returns a DataFrame matching ONCHAIN_TRADE_SCHEMA.
    """
    files = sorted(Path(onchain_dir).glob("order_filled_*.parquet"))
    if not files:
        return pl.DataFrame(schema=ONCHAIN_TRADE_SCHEMA)

    raw = pl.concat([pl.read_parquet(f) for f in files])
    wanted_tokens: list[str] = []
    if yes_token_id:
        wanted_tokens.append(yes_token_id)
    if no_token_id:
        wanted_tokens.append(no_token_id)
    if not wanted_tokens:
        return pl.DataFrame(schema=ONCHAIN_TRADE_SCHEMA)

    # aggregate_onchain_df already handles the USDC-sign decode.
    agg = aggregate_onchain_df(raw)
    agg = agg.filter(pl.col("token_id").is_in(wanted_tokens))
    if agg.height == 0:
        return pl.DataFrame(schema=ONCHAIN_TRADE_SCHEMA)

    # Attach block timestamps via linear interpolation from anchor block.
    agg = _attach_block_ts_linear(agg, rpc_url)

    if t_start is not None:
        agg = agg.filter(pl.col("t") >= t_start)
    if t_end is not None:
        agg = agg.filter(pl.col("t") < t_end)

    # Carry maker/taker through from raw (first within each bucket).
    pre = raw.select([
        "block_number", "tx_hash", "maker", "taker",
    ]).unique(subset=["block_number", "tx_hash"])
    agg = agg.join(pre, on=["block_number", "tx_hash"], how="left")

    return agg.with_columns(
        pl.lit(market_id).alias("market_id"),
    ).select(
        ["market_id", "t", "block_number", "tx_hash", "token_id",
         "price", "size", "sign", "maker", "taker"],
    ).with_columns(
        [pl.col(k).cast(v) for k, v in ONCHAIN_TRADE_SCHEMA.items()],
    )


class OnchainTradeStream:
    """Iterator analog of `MarketStream` for authoritative on-chain trades.

    Loads `load_onchain_trades` output once, sorts by ts, yields
    `OnchainTrade` dataclass instances. Useful for trade-level streaming
    consumers (Plan 3c stylized-fact work that processes trades one-by-one).

    For measure injection, prefer `load_onchain_trades` directly.
    """

    def __init__(
        self,
        market_id: str,
        yes_token_id: str | None,
        no_token_id: str | None,
        t_start: datetime | None = None,
        t_end: datetime | None = None,
        onchain_dir: Path = Path("data/onchain_fills"),
        rpc_url: str = "https://polygon-rpc.com",
    ) -> None:
        self.market_id = market_id
        self._df = load_onchain_trades(
            market_id=market_id,
            yes_token_id=yes_token_id, no_token_id=no_token_id,
            t_start=t_start, t_end=t_end,
            onchain_dir=onchain_dir, rpc_url=rpc_url,
        ).sort("t")

    def __iter__(self) -> Iterator[OnchainTrade]:
        for row in self._df.iter_rows(named=True):
            yield OnchainTrade(
                ts=row["t"],
                block_number=int(row["block_number"]),
                tx_hash=row["tx_hash"],
                token_id=row["token_id"],
                price=Decimal(str(row["price"])),
                size=Decimal(str(row["size"])),
                sign=int(row["sign"]),
                maker=row["maker"] or "",
                taker=row["taker"] or "",
            )

    def __len__(self) -> int:
        return self._df.height


from polydata.resample import LOBSample  # noqa: E402


def attach_pre_trade_mid(
    trades: pl.DataFrame, samples: list[LOBSample],
) -> pl.DataFrame:
    """Attach `mid_at_trade` column — LOB mid at the latest sample whose
    timestamp is at or before each trade's `t`.

    If no prior sample exists (trade before first sample), `mid_at_trade`
    is null. Samples must be sorted by `.t` ascending.
    """
    if trades.height == 0 or not samples:
        return trades.with_columns(pl.lit(None, dtype=pl.Float64).alias("mid_at_trade"))

    sample_ts = [s.t for s in samples]
    sample_mids = [
        (float(s.best_bid) + float(s.best_ask)) / 2
        if (s.best_bid is not None and s.best_ask is not None) else None
        for s in samples
    ]
    samples_df = pl.DataFrame(
        {"sample_t": sample_ts, "mid_at_trade": sample_mids},
        schema={
            "sample_t": pl.Datetime(time_zone="UTC"),
            "mid_at_trade": pl.Float64,
        },
    ).sort("sample_t")

    return trades.sort("t").join_asof(
        samples_df, left_on="t", right_on="sample_t", strategy="backward",
    ).drop("sample_t")
