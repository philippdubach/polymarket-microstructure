"""Binary-search a Polygon block number for a given UTC timestamp.

Usage:
    uv run python scripts/find_block_at_timestamp.py 2026-02-28T00:00:00Z
    uv run python scripts/find_block_at_timestamp.py 2026-03-28T00:00:00Z

Prints the block number whose timestamp is the LAST <= the target.
Uses POLYGON_RPC_URL env var, falls back to public polygon-rpc.com.
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

from polydata.onchain.rpc import PolygonRpcClient


def find_block_at_or_before(client: PolygonRpcClient, target_unix: int) -> int:
    lo = 1
    hi = client.block_number()
    # Binary search: largest block with timestamp <= target.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ts = client.block_timestamp(mid)
        if ts <= target_unix:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: find_block_at_timestamp.py <ISO8601 UTC timestamp>")
        return 2
    iso = sys.argv[1].replace("Z", "+00:00")
    target = datetime.fromisoformat(iso)
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    target_unix = int(target.timestamp())
    url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    with PolygonRpcClient(url) as client:
        block = find_block_at_or_before(client, target_unix)
        ts = client.block_timestamp(block)
        resolved = datetime.fromtimestamp(ts, tz=UTC)
    print(f"{block}\t# {resolved.isoformat()}  (target {target.isoformat()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
