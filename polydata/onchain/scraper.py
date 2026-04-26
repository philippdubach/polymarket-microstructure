"""OrderFilled scraper clamped to finalized depth."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from polydata.onchain.events import (
    ORDER_FILLED_TOPIC,
    OrderFilledEvent,
    decode_order_filled,
)
from polydata.onchain.rpc import PolygonRpcClient

FINALITY_BUFFER_BLOCKS: int = 256

_SCHEMA: dict = {
    "block_number": pl.UInt64,
    "tx_hash": pl.Utf8,
    "log_index": pl.UInt32,
    "order_hash": pl.Utf8,
    "maker": pl.Utf8,
    "taker": pl.Utf8,
    "maker_asset_id": pl.Utf8,
    "taker_asset_id": pl.Utf8,
    "maker_amount_filled": pl.Utf8,
    "taker_amount_filled": pl.Utf8,
    "fee": pl.Utf8,
}


def _event_to_row(ev: OrderFilledEvent) -> dict:
    return {
        "block_number": ev.block_number,
        "tx_hash": ev.tx_hash,
        "log_index": ev.log_index,
        "order_hash": ev.order_hash,
        "maker": ev.maker,
        "taker": ev.taker,
        "maker_asset_id": str(ev.maker_asset_id),
        "taker_asset_id": str(ev.taker_asset_id),
        "maker_amount_filled": str(ev.maker_amount_filled),
        "taker_amount_filled": str(ev.taker_amount_filled),
        "fee": str(ev.fee),
    }


def scrape_order_filled(
    client: PolygonRpcClient,
    contract_address: str,
    from_block: int,
    to_block: int,
    out_path: Path,
    chunk: int = 1000,
) -> None:
    rows = [
        _event_to_row(decode_order_filled(log))
        for log in client.get_logs(
            address=contract_address,
            topics=[ORDER_FILLED_TOPIC],
            from_block=from_block,
            to_block=to_block,
            chunk=chunk,
        )
    ]
    df = pl.DataFrame(rows, schema=_SCHEMA) if rows else pl.DataFrame(schema=_SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)


def scrape_slice_finalized(
    client: PolygonRpcClient,
    contract_address: str,
    from_block: int,
    requested_to_block: int,
    out_dir: Path,
    slice_blocks: int = 100_000,
    chunk: int = 1000,
) -> list[Path]:
    """Scrape [from_block, min(requested_to_block, finalized - 256)] in slices.

    Idempotent: skips slices whose output parquet already exists.
    """
    finalized = client.finalized_block_number()
    safe_to_block = min(requested_to_block, finalized - FINALITY_BUFFER_BLOCKS)
    if safe_to_block < from_block:
        raise RuntimeError(
            f"requested range {from_block}..{requested_to_block} is ahead of "
            f"finalized-{FINALITY_BUFFER_BLOCKS} = {safe_to_block}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    start = from_block
    while start <= safe_to_block:
        end = min(start + slice_blocks - 1, safe_to_block)
        out_path = out_dir / f"order_filled_{start}_{end}.parquet"
        if out_path.exists():
            produced.append(out_path)
            start = end + 1
            continue
        scrape_order_filled(
            client=client,
            contract_address=contract_address,
            from_block=start,
            to_block=end,
            out_path=out_path,
            chunk=chunk,
        )
        produced.append(out_path)
        start = end + 1
    return produced
