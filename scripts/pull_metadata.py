"""One-off: enumerate distinct market_ids in the archive, pull metadata, cache."""
import polars as pl

from polydata.metadata import fetch_metadata, save_cache
from polydata.paths import parquet_files


def enumerate_markets() -> list[str]:
    lfs = [pl.scan_parquet(f).select("market_id").unique() for f in parquet_files()]
    combined = pl.concat(lfs).unique().collect(streaming=True)
    return combined["market_id"].to_list()


if __name__ == "__main__":
    ids = enumerate_markets()
    print(f"distinct market_ids: {len(ids)}")
    md = fetch_metadata(ids)
    save_cache(md)
    print(f"cached metadata rows: {len(md)}")
