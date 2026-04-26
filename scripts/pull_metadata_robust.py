"""Robust Gamma API metadata pull — run in a foreground stable shell.

Differences from scripts/pull_metadata.py:
- Enumerates distinct market_ids via DuckDB instead of Polars streaming concat
  (handles 1,262 parquet files without OOM).
- Writes a checkpoint parquet of the enumerated ids before hitting the API.
- Batches API calls with retry/backoff and progress printing every N requests.
- Saves the metadata cache incrementally (every 1,000 markets) so a crash
  does not lose prior work.
- Unbuffered stdout so progress shows live (`python -u` or explicit flush).

Usage (from a terminal, not a Claude session):
    cd /Users/philippdubach/Desktop/poly-data-paper
    uv run python -u scripts/pull_metadata_robust.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import httpx
import polars as pl

from polydata.metadata import parse_gamma_response
from polydata.paths import DATA_DIR, METADATA_CACHE, parquet_files

CHECKPOINT_IDS = METADATA_CACHE.parent / "market_ids_checkpoint.parquet"
INCREMENTAL_CACHE = METADATA_CACHE.parent / "metadata_cache.partial.parquet"
GAMMA_URL = "https://gamma-api.polymarket.com/markets"

BATCH_SIZE = 100
PROGRESS_EVERY = 20       # log every N API calls
SAVE_EVERY = 1000         # incremental save every N markets fetched
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def enumerate_market_ids_duckdb() -> list[str]:
    """Collect distinct market_ids across ALL parquet files via DuckDB.

    DuckDB streams cross-file aggregation without loading everything into
    memory. Returns a sorted list.
    """
    log("enumerating distinct market_ids via DuckDB…")
    glob = str(Path(DATA_DIR) / "polymarket_orderbook_*.parquet")
    con = duckdb.connect()
    q = f"SELECT DISTINCT market_id FROM read_parquet('{glob}') ORDER BY market_id"
    start = time.time()
    ids = [row[0] for row in con.execute(q).fetchall()]
    elapsed = time.time() - start
    log(f"found {len(ids):,} distinct market_ids in {elapsed:.1f}s")
    return ids


def load_checkpoint_ids() -> list[str] | None:
    if CHECKPOINT_IDS.exists():
        df = pl.read_parquet(CHECKPOINT_IDS)
        ids = df["market_id"].to_list()
        log(f"loaded {len(ids):,} ids from checkpoint {CHECKPOINT_IDS.name}")
        return ids
    return None


def save_checkpoint_ids(ids: list[str]) -> None:
    CHECKPOINT_IDS.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"market_id": ids}).write_parquet(CHECKPOINT_IDS)
    log(f"saved ids checkpoint to {CHECKPOINT_IDS}")


def load_partial_cache() -> set[str]:
    if INCREMENTAL_CACHE.exists():
        df = pl.read_parquet(INCREMENTAL_CACHE)
        done = set(df["market_id"].to_list())
        log(f"loaded {len(done):,} already-fetched markets from partial cache")
        return done
    return set()


def save_partial_cache(records: list) -> None:
    if not records:
        return
    df = pl.DataFrame([r.__dict__ for r in records])
    df.write_parquet(INCREMENTAL_CACHE)


def fetch_with_retry(client: httpx.Client, params) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.get(GAMMA_URL, params=params, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.RequestError) as e:
            wait = BACKOFF_BASE ** attempt
            cls = e.__class__.__name__
            log(f"  retry {attempt}/{MAX_RETRIES} after {cls}: {e}. sleep {wait:.1f}s")
            time.sleep(wait)
    log(f"  GAVE UP on batch after {MAX_RETRIES} retries")
    return []


def main() -> int:
    log(f"data dir: {DATA_DIR}")
    log(f"cache target: {METADATA_CACHE}")

    # Step 1: enumerate market_ids (with checkpointing)
    ids = load_checkpoint_ids()
    if ids is None:
        files = parquet_files()
        log(f"{len(files)} parquet files on disk")
        ids = enumerate_market_ids_duckdb()
        save_checkpoint_ids(ids)
    total = len(ids)

    # Step 2: skip already-fetched markets
    already = load_partial_cache()
    to_fetch = [m for m in ids if m not in already]
    log(f"to fetch: {len(to_fetch):,} / {total:,} ({len(already):,} already cached)")

    if not to_fetch:
        log("nothing to fetch — promoting partial cache to final")
        if INCREMENTAL_CACHE.exists():
            INCREMENTAL_CACHE.rename(METADATA_CACHE)
            log(f"renamed partial → {METADATA_CACHE}")
        return 0

    # Step 3: batched API pull
    new_records = []
    if INCREMENTAL_CACHE.exists():
        prev = pl.read_parquet(INCREMENTAL_CACHE)
        prev_records = prev.to_dicts()
    else:
        prev_records = []

    fetched_count = 0
    api_calls = 0
    batches = [to_fetch[i:i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
    log(f"{len(batches)} batches of up to {BATCH_SIZE}")

    with httpx.Client(timeout=30.0) as client:
        for batch_i, batch in enumerate(batches, start=1):
            params = tuple(("condition_ids", m) for m in batch)
            rows = fetch_with_retry(client, params)
            api_calls += 1
            for row in rows:
                md = parse_gamma_response(row)
                new_records.append(md)
                fetched_count += 1

            if api_calls % PROGRESS_EVERY == 0:
                pct = 100 * batch_i / len(batches)
                log(f"batch {batch_i}/{len(batches)} ({pct:.1f}%) · {fetched_count:,} new records")

            if fetched_count and fetched_count % SAVE_EVERY == 0:
                # Incremental save
                combined = prev_records + [r.__dict__ for r in new_records]
                pl.DataFrame(combined).write_parquet(INCREMENTAL_CACHE)
                log(f"  checkpoint: {len(combined):,} markets saved to partial cache")

            # Polite pacing
            time.sleep(0.1)

    # Step 4: final save
    combined = prev_records + [r.__dict__ for r in new_records]
    log(f"final: {len(combined):,} records total")
    if combined:
        pl.DataFrame(combined).write_parquet(METADATA_CACHE)
        log(f"wrote {METADATA_CACHE}")
        # Clean up partial
        if INCREMENTAL_CACHE.exists():
            INCREMENTAL_CACHE.unlink()
    else:
        log("no records fetched; cache unchanged")

    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
