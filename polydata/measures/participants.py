"""Market-maker-like activity signal.

Without wallet identifiers in the price_change feed, we cannot identify
individual MMs. We instead emit a per-window signal that aggregates
MM-LIKE behaviors — high cancel share, low latency, high quote churn —
as a proxy for MM presence.

Output columns:
- cancel_share: fraction of price_change events with change_size == 0
- fast_share_p50ms: fraction of events whose latency is below
  FAST_LATENCY_MS_THRESHOLD (100 ms by default; Plan 3 will calibrate
  per-market)
- median_latency_ms: median latency across events in the window
- quote_churn_per_sec: n_events / window_duration_seconds
"""
from __future__ import annotations

import numpy as np
import polars as pl

from polydata.events import PriceChange
from polydata.window import MeasurementWindow

FAST_LATENCY_MS_THRESHOLD = 100.0


def mm_activity_signal(window: MeasurementWindow) -> pl.DataFrame:
    n_events = 0
    n_cancels = 0
    n_fast = 0
    latencies_ms: list[float] = []
    for r in window.events:
        if not isinstance(r.event, PriceChange):
            continue
        n_events += 1
        if r.event.change_size == 0:
            n_cancels += 1
        lat = (r.ts_created - r.ts_received).total_seconds() * 1000.0
        latencies_ms.append(lat)
        if lat <= FAST_LATENCY_MS_THRESHOLD:
            n_fast += 1
    duration_s = max(window.duration().total_seconds(), 1e-9)
    if n_events == 0:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_events": 0,
            "cancel_share": 0.0, "fast_share_p50ms": 0.0,
            "median_latency_ms": 0.0, "quote_churn_per_sec": 0.0,
        }], schema={
            "market_id": pl.Utf8, "n_events": pl.UInt32,
            "cancel_share": pl.Float64, "fast_share_p50ms": pl.Float64,
            "median_latency_ms": pl.Float64, "quote_churn_per_sec": pl.Float64,
        })
    return pl.DataFrame([{
        "market_id": window.market_id,
        "n_events": int(n_events),
        "cancel_share": n_cancels / n_events,
        "fast_share_p50ms": n_fast / n_events,
        "median_latency_ms": float(np.median(latencies_ms)),
        "quote_churn_per_sec": n_events / duration_s,
    }], schema={
        "market_id": pl.Utf8, "n_events": pl.UInt32,
        "cancel_share": pl.Float64, "fast_share_p50ms": pl.Float64,
        "median_latency_ms": pl.Float64, "quote_churn_per_sec": pl.Float64,
    })
