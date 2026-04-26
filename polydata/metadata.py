from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl

from polydata.paths import METADATA_CACHE

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


@dataclass(frozen=True)
class MarketMetadata:
    market_id: str
    question: str
    category: str | None
    end_date: datetime | None
    closed_time: datetime | None
    resolved_outcome: int | None  # 0 = YES won, 1 = NO won, None = unresolved


def _parse_dt(v: str | None) -> datetime | None:
    if not v:
        return None
    return datetime.fromisoformat(v.replace("Z", "+00:00"))


def parse_gamma_response(d: dict) -> MarketMetadata:
    prices = d.get("outcomePrices") or []
    resolved = None
    if d.get("closed") and len(prices) == 2:
        try:
            p0, p1 = float(prices[0]), float(prices[1])
            resolved = 0 if p0 > p1 else 1
        except (TypeError, ValueError):
            resolved = None
    return MarketMetadata(
        market_id=d["conditionId"],
        question=d.get("question", ""),
        category=d.get("category"),
        end_date=_parse_dt(d.get("endDate")),
        closed_time=_parse_dt(d.get("closedTime")),
        resolved_outcome=resolved,
    )


def fetch_metadata(market_ids: Iterable[str], batch_size: int = 100) -> list[MarketMetadata]:
    ids = list(market_ids)
    out: list[MarketMetadata] = []
    with httpx.Client(timeout=30) as client:
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            params = tuple(("condition_ids", m) for m in batch)
            r = client.get(GAMMA_MARKETS_URL, params=params)
            r.raise_for_status()
            for row in r.json():
                out.append(parse_gamma_response(row))
    return out


def save_cache(records: list[MarketMetadata], path: Path = METADATA_CACHE) -> None:
    df = pl.DataFrame([r.__dict__ for r in records])
    df.write_parquet(path)


def load_cache(path: Path = METADATA_CACHE) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(
            schema={
                "market_id": pl.Utf8,
                "question": pl.Utf8,
                "category": pl.Utf8,
                "end_date": pl.Datetime,
                "closed_time": pl.Datetime,
                "resolved_outcome": pl.Int64,
            }
        )
    return pl.read_parquet(path)
