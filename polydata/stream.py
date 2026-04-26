from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from polydata.events import Event, parse_event
from polydata.paths import parquet_files


@dataclass(frozen=True)
class StreamRecord:
    ts_received: datetime
    ts_created: datetime
    event: Event


class MarketStream:
    def __init__(
        self,
        market_id: str,
        files: list[Path] | None = None,
        side: str | None = None,
    ) -> None:
        self.market_id = market_id
        self.files = files if files is not None else parquet_files()
        self.side = side

    def __iter__(self) -> Iterator[StreamRecord]:
        for f in self.files:
            lf = pl.scan_parquet(f).filter(pl.col("market_id") == self.market_id)
            df = lf.sort("timestamp_received").collect(engine="streaming")
            for row in df.iter_rows(named=True):
                ev = parse_event(row["data"])
                if self.side is not None and ev.side != self.side:
                    continue
                yield StreamRecord(
                    ts_received=row["timestamp_received"],
                    ts_created=row["timestamp_created_at"],
                    event=ev,
                )
