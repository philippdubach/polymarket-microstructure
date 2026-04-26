"""One-off: pull clob_token_ids for all markets already in metadata cache.

Extends Plan 1 Task 10's metadata cache with the YES/NO CTF token ids.
Writes to data/clob_token_map.parquet.
"""

from __future__ import annotations

import polars as pl

from polydata.metadata import load_cache as load_metadata_cache
from polydata.onchain.token_map import (
    fetch_clob_token_ids_bulk,
    load_token_map,
    save_token_map,
)


def main() -> int:
    meta = load_metadata_cache()
    if meta.height == 0:
        print("metadata cache is empty; run scripts/pull_metadata_robust.py first")
        return 1
    existing = load_token_map()
    missing = set(meta["market_id"].to_list()) - set(existing["market_id"].to_list())
    print(f"metadata markets: {meta.height:,}  missing token ids: {len(missing):,}")
    if not missing:
        print("already complete")
        return 0
    new_rows = fetch_clob_token_ids_bulk(sorted(missing))
    combined = pl.concat([existing, new_rows]) if existing.height > 0 else new_rows
    save_token_map(combined)
    print(f"wrote {combined.height:,} rows to data/clob_token_map.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
