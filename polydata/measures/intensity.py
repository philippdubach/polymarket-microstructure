"""Event-intensity measures.

- quote_update_intensity: count of price_change events per bucket.
- cancel_to_fill_ratio: proxy for spoofing. A cancel is defined as a
  price_change with change_size == 0 (level removal). A fill is detected
  as a decrement in resting size at a level from one event to the next at
  the same price + side — in this quote-only feed we approximate fills as
  "price_change where the new change_size is strictly less than the size
  at the same (price, side) immediately before." Plan 2b (trade inference)
  will give a more accurate fill count; this is the quote-side lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot, PriceChange
from polydata.window import MeasurementWindow


def quote_update_intensity(
    window: MeasurementWindow,
    bucket: timedelta = timedelta(seconds=1),
) -> pl.DataFrame:
    ts = [r.ts_received for r in window.events
          if isinstance(r.event, PriceChange)]
    if not ts:
        return pl.DataFrame(schema={
            "market_id": pl.Utf8,
            "bucket_t": pl.Datetime(time_zone="UTC"),
            "n_updates": pl.UInt32,
        })
    df = pl.DataFrame({"t": ts}).with_columns(
        pl.col("t").dt.truncate(bucket).alias("bucket_t"),
    )
    out = df.group_by("bucket_t").agg(pl.len().alias("n_updates")).sort("bucket_t")
    return out.with_columns(pl.lit(window.market_id).alias("market_id")).select(
        ["market_id", "bucket_t", "n_updates"]
    )


@dataclass(frozen=True)
class CancelFillReport:
    n_cancels: int
    n_fills: int
    ratio: float  # n_cancels / n_fills; math.inf if n_fills == 0


def cancel_to_fill_ratio(window: MeasurementWindow) -> CancelFillReport:
    n_cancels = 0
    n_fills = 0
    last_size_at: dict[tuple[Decimal, str], Decimal] = {}
    for rec in window.events:
        ev = rec.event
        if isinstance(ev, BookSnapshot):
            last_size_at.clear()
            for p, q in ev.bids:
                last_size_at[(p, "BUY")] = q
            for p, q in ev.asks:
                last_size_at[(p, "SELL")] = q
            continue
        if not isinstance(ev, PriceChange):
            continue
        key = (ev.change_price, ev.change_side)
        prev = last_size_at.get(key, Decimal("0"))
        if ev.change_size == 0:
            n_cancels += 1
        elif ev.change_size < prev:
            n_fills += 1
        last_size_at[key] = ev.change_size
    ratio = math.inf if n_fills == 0 else n_cancels / n_fills
    return CancelFillReport(n_cancels=n_cancels, n_fills=n_fills, ratio=ratio)
