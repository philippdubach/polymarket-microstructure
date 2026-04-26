from pathlib import Path

DATA_DIR = Path("/Volumes/data/polymarket")
METADATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "metadata_cache.parquet"
METADATA_CACHE.parent.mkdir(parents=True, exist_ok=True)

def parquet_files() -> list[Path]:
    return sorted(DATA_DIR.glob("polymarket_orderbook_*.parquet"))
