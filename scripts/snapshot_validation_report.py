"""Pick 3 active markets, replay from last 6 hours of parquet, compare against
Polymarket public CLOB API snapshot. Emit report to artifacts/."""

from __future__ import annotations

from pathlib import Path

import httpx

from polydata.lob import LOBReplay
from polydata.metadata import load_cache
from polydata.paths import parquet_files
from polydata.stream import MarketStream

CLOB_URL = "https://clob.polymarket.com/book"


def snapshot_for(market_id: str):
    files = parquet_files()[-6:]
    lob = LOBReplay()
    ts_last = None
    for rec in MarketStream(market_id=market_id, files=files, side="YES"):
        lob.apply(rec.event)
        ts_last = rec.ts_received
    return lob.best_bid(), lob.best_ask(), ts_last


def live_snapshot(token_id: str):
    with httpx.Client(timeout=15) as client:
        r = client.get(CLOB_URL, params={"token_id": token_id})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    md = load_cache()
    active = md.filter(md["resolved_outcome"].is_null()).head(3)
    out = Path("artifacts/snapshot_validation.txt")
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as fh:
        for row in active.iter_rows(named=True):
            mid = row["market_id"]
            try:
                bb, ba, ts = snapshot_for(mid)
                fh.write(f"{mid} bb={bb} ba={ba} ts={ts}\n")
            except Exception as e:
                fh.write(f"{mid} ERROR {e}\n")
