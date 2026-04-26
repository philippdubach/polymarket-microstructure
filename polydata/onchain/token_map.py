"""market_id (conditionId) → (yes_token_id, no_token_id) mapping.

Polymarket Gamma API returns `clobTokenIds` as a JSON-encoded array of
two strings: [yes_id, no_id]. We cache the mapping in a parquet file
(`data/clob_token_map.parquet`) to avoid re-querying Gamma.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import httpx
import polars as pl

from polydata.paths import METADATA_CACHE

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
TOKEN_MAP_CACHE = METADATA_CACHE.parent / "clob_token_map.parquet"

TOKEN_MAP_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "yes_token_id": pl.Utf8,
    "no_token_id": pl.Utf8,
}


def parse_gamma_clob_token_ids(row: dict) -> tuple[str | None, str | None]:
    raw = row.get("clobTokenIds")
    if not raw:
        return None, None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None, None
    return str(parsed[0]), str(parsed[1])


def fetch_clob_token_ids_bulk(
    market_ids: Iterable[str],
    batch_size: int = 100,
) -> pl.DataFrame:
    ids = list(market_ids)
    rows: list[dict] = []
    with httpx.Client(timeout=30) as client:
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            params = tuple(("condition_ids", m) for m in batch)
            r = client.get(GAMMA_URL, params=params)
            r.raise_for_status()
            for gamma_row in r.json():
                yes, no = parse_gamma_clob_token_ids(gamma_row)
                rows.append(
                    {
                        "market_id": gamma_row.get("conditionId", ""),
                        "yes_token_id": yes,
                        "no_token_id": no,
                    }
                )
    return (
        pl.DataFrame(rows, schema=TOKEN_MAP_SCHEMA)
        if rows
        else pl.DataFrame(schema=TOKEN_MAP_SCHEMA)
    )


def save_token_map(df: pl.DataFrame, path: Path = TOKEN_MAP_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def load_token_map(path: Path = TOKEN_MAP_CACHE) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=TOKEN_MAP_SCHEMA)
    return pl.read_parquet(path)


def resolve_token_ids(
    market_id: str,
    cache: pl.DataFrame,
) -> tuple[str | None, str | None]:
    row = cache.filter(pl.col("market_id") == market_id)
    if row.height == 0:
        return None, None
    r = row.row(0, named=True)
    return r["yes_token_id"], r["no_token_id"]
