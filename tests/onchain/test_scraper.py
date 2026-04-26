# tests/onchain/test_scraper.py
import tempfile
from pathlib import Path

import httpx
import polars as pl

from polydata.onchain.events import ORDER_FILLED_TOPIC
from polydata.onchain.rpc import PolygonRpcClient
from polydata.onchain.scraper import scrape_order_filled, scrape_slice_finalized


def _fake_log(block: int, log_idx: int, tx: str) -> dict:
    data = (
        "0x"
        + "00" * 31 + "01" + "00" * 31 + "02"
        + "00" * 30 + "03e8" + "00" * 30 + "01f4"
        + "00" * 32
    )
    return {
        "blockNumber": hex(block), "transactionHash": tx, "logIndex": hex(log_idx),
        "topics": [
            ORDER_FILLED_TOPIC,
            "0x" + "aa" * 32,
            "0x" + "00" * 12 + "bb" * 20,
            "0x" + "00" * 12 + "cc" * 20,
        ],
        "data": data,
    }


class MockTransport(httpx.BaseTransport):
    def __init__(self, finalized_block, logs_chunks):
        self.finalized_block = finalized_block
        self.logs_chunks = list(logs_chunks)

    def handle_request(self, request):
        body = request.content.decode()
        if "eth_getBlockByNumber" in body and "finalized" in body:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1,
                "result": {"number": hex(self.finalized_block), "timestamp": "0x0"}})
        if "eth_getLogs" in body:
            next_chunk = self.logs_chunks.pop(0) if self.logs_chunks else []
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": next_chunk})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x0"})


def test_scrape_writes_parquet_with_schema():
    # NOTE: get_logs breaks on empty result, so all chunks must be non-empty.
    # Use chunk=1000 with from_block=0/to_block=2999 → exactly 3 iterations
    # (0-999, 1000-1999, 2000-2999), each returning one log.
    transport = MockTransport(finalized_block=5000,
                              logs_chunks=[[_fake_log(100, 0, "0x" + "1" * 64)],
                                           [_fake_log(1100, 0, "0x" + "3" * 64)],
                                           [_fake_log(2100, 0, "0x" + "2" * 64)]])
    client = PolygonRpcClient("https://mock/rpc", transport=transport)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fills.parquet"
        scrape_order_filled(
            client=client, contract_address="0xExch",
            from_block=0, to_block=2999, chunk=1000, out_path=out,
        )
        df = pl.read_parquet(out)
        assert df.height == 3
        assert set(df.columns) >= {"block_number", "tx_hash", "log_index",
            "order_hash", "maker", "taker", "maker_asset_id", "taker_asset_id",
            "maker_amount_filled", "taker_amount_filled", "fee"}
        # verify all 3 block numbers are present
        assert set(df["block_number"].to_list()) == {100, 1100, 2100}


def test_scrape_slice_finalized_clamps_to_finalized_block():
    logs = [_fake_log(100, 0, "0x" + "1" * 64)]
    transport = MockTransport(finalized_block=500, logs_chunks=[logs])
    client = PolygonRpcClient("https://mock/rpc", transport=transport)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "slices"
        produced = scrape_slice_finalized(
            client=client, contract_address="0xExch",
            from_block=100, requested_to_block=2000,
            out_dir=out_dir, slice_blocks=1000, chunk=1000,
        )
        # finalized is 500 — scraper should clamp to_block at 500−256 = 244
        assert len(produced) == 1
        df = pl.read_parquet(produced[0])
        assert "block_number" in df.columns
