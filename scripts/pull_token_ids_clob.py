"""Paginate GET clob.polymarket.com/markets to extract (condition_id,
yes_token_id, no_token_id) for every market CLOB knows about.

Replaces the Gamma-based `pull_token_ids.py` which only resolves ~6,960 of
the 34,764 metadata-cached markets. CLOB's bulk endpoint returns ALL
markets (active + closed + archived) with their token pair directly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import polars as pl

from polydata.onchain.token_map import TOKEN_MAP_CACHE, TOKEN_MAP_SCHEMA

CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"
PAGE_LIMIT = 1000
REQ_SLEEP_S = 0.2  # 5 req/sec — well under any sane rate limit


def _extract_yes_no(market_row: dict) -> tuple[str | None, str | None]:
    tokens = market_row.get("tokens") or []
    yes_id, no_id = None, None
    for t in tokens:
        outcome = (t.get("outcome") or "").strip().lower()
        tid = t.get("token_id")
        if not tid:
            continue
        if outcome == "yes":
            yes_id = tid
        elif outcome == "no":
            no_id = tid
    # fallback: if outcome labels missing but exactly 2 tokens, index-order
    if yes_id is None and no_id is None and len(tokens) == 2:
        yes_id = tokens[0].get("token_id")
        no_id = tokens[1].get("token_id")
    return yes_id, no_id


def main() -> int:
    rows: list[dict] = []
    cursor = ""
    page = 0
    with httpx.Client(timeout=30) as client:
        while True:
            params = {"next_cursor": cursor} if cursor else {}
            r = client.get(CLOB_MARKETS_URL, params=params)
            r.raise_for_status()
            body = r.json()
            data = body.get("data") or []
            for mkt in data:
                cid = mkt.get("condition_id")
                if not cid:
                    continue
                y, n = _extract_yes_no(mkt)
                rows.append(
                    {"market_id": cid, "yes_token_id": y, "no_token_id": n}
                )
            cursor = body.get("next_cursor") or ""
            page += 1
            print(
                f"page {page}: +{len(data)} markets  "
                f"total={len(rows):,}  next_cursor={cursor[:20] or '(none)'}",
                flush=True,
            )
            if not cursor or cursor == "LTE=":
                break
            time.sleep(REQ_SLEEP_S)

    if not rows:
        print("no rows fetched", file=sys.stderr)
        return 1

    df = (
        pl.DataFrame(rows, schema=TOKEN_MAP_SCHEMA)
        .unique(subset=["market_id"])
        .sort("market_id")
    )
    Path(TOKEN_MAP_CACHE).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(TOKEN_MAP_CACHE)
    n_with_yes = int(df["yes_token_id"].is_not_null().sum())
    n_with_no = int(df["no_token_id"].is_not_null().sum())
    print(
        f"\nwrote {df.height:,} rows to {TOKEN_MAP_CACHE}  "
        f"(yes_id: {n_with_yes:,}  no_id: {n_with_no:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
