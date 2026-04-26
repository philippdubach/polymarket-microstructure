"""Long-running scraper. Env vars:
    POLYGON_RPC_URL, CTF_EXCHANGE_ADDRESS,
    CALIB_START_BLOCK, CALIB_END_BLOCK,
    SCRAPE_CHUNK (default 200 — per-`eth_getLogs` block span;
        free-tier limits: Alchemy 10, Ankr 100, PublicNode ~1000).
    SCRAPE_SLICE_BLOCKS (default 100000 — per-parquet checkpoint span;
        smaller = more frequent writes but more metadata overhead).
"""
from __future__ import annotations

import os
from pathlib import Path

from polydata.onchain.rpc import PolygonRpcClient
from polydata.onchain.scraper import scrape_slice_finalized


def main() -> int:
    url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    addr = os.environ["CTF_EXCHANGE_ADDRESS"]
    start = int(os.environ["CALIB_START_BLOCK"])
    end = int(os.environ["CALIB_END_BLOCK"])
    chunk = int(os.environ.get("SCRAPE_CHUNK", "200"))
    slice_blocks = int(os.environ.get("SCRAPE_SLICE_BLOCKS", "100000"))
    out_dir = Path("data/onchain_fills")
    print(f"scraping {addr} from {start} to {end} "
          f"(clamped to finalized-256, chunk={chunk}, slice={slice_blocks}) → {out_dir}")
    with PolygonRpcClient(url) as client:
        produced = scrape_slice_finalized(
            client=client, contract_address=addr,
            from_block=start, requested_to_block=end,
            out_dir=out_dir, slice_blocks=slice_blocks, chunk=chunk,
        )
    print(f"wrote {len(produced)} slice parquets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
