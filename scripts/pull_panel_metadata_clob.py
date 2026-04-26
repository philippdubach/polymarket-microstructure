"""Pull per-market metadata from CLOB REST for the panel.

CLOB REST /markets/{conditionId} returns rich metadata including
`question`, `end_date_iso`, `market_slug`, `closed`. We use this to
unblock SF5 (category via keyword on question) and SF8 (depth decay
needs `end_date_iso` for time-to-resolution).

Output: data/panel_metadata.parquet with one row per panel market.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import polars as pl

CLOB_MARKET_URL = "https://clob.polymarket.com/markets/{cid}"
SLEEP_S = 0.15  # ~6 req/sec — well below any sane rate limit


def main() -> int:
    panel = pl.read_parquet("data/panel.parquet")
    market_ids = panel["market_id"].to_list()
    print(f"pulling metadata for {len(market_ids)} panel markets…")
    rows: list[dict] = []
    failed: list[str] = []
    with httpx.Client(timeout=30) as client:
        for i, mid in enumerate(market_ids, start=1):
            try:
                r = client.get(CLOB_MARKET_URL.format(cid=mid))
                if r.status_code != 200:
                    failed.append(mid)
                    continue
                data = r.json()
                rows.append({
                    "market_id": data.get("condition_id", mid),
                    "question": data.get("question"),
                    "market_slug": data.get("market_slug"),
                    "end_date_iso": data.get("end_date_iso"),
                    "closed": data.get("closed", False),
                    "active": data.get("active", False),
                    "tags": ",".join(data.get("tags") or []),
                })
            except Exception as e:
                print(f"  [{i}/{len(market_ids)}] {mid[:14]}… ERROR {e}")
                failed.append(mid)
            if i % 50 == 0:
                print(
                    f"  [{i}/{len(market_ids)}] hits={len(rows)}  "
                    f"failed={len(failed)}",
                    flush=True,
                )
            time.sleep(SLEEP_S)
    print(f"\nfetched: {len(rows)}, failed: {len(failed)}")
    if not rows:
        print("no rows fetched")
        return 1
    df = pl.DataFrame(rows)
    out = Path("data/panel_metadata.parquet")
    df.write_parquet(out)
    print(f"wrote {df.height:,} rows to {out}")
    if failed:
        print(f"first 5 failures: {failed[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
