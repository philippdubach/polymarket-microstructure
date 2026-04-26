"""Two-clock latency distribution.

latency = ts_created_at - ts_received, expressed in milliseconds.
`ts_received` is the earlier clock — the exchange-side event time.
`ts_created_at` is the later clock — when the archive collector created
the record. The positive difference is propagation + archive-processing
delay. Unique to this dataset — no prior published Polymarket paper has
access to both clocks.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from polydata.window import MeasurementWindow


def latency_distribution(window: MeasurementWindow) -> pl.DataFrame:
    ms = [
        (r.ts_created - r.ts_received).total_seconds() * 1000.0
        for r in window.events
    ]
    if not ms:
        return pl.DataFrame(schema={
            "market_id": pl.Utf8, "n": pl.UInt32,
            "p50_ms": pl.Float64, "p90_ms": pl.Float64, "p99_ms": pl.Float64,
            "mean_ms": pl.Float64, "max_ms": pl.Float64,
        })
    arr = np.array(ms)
    return pl.DataFrame([{
        "market_id": window.market_id,
        "n": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }], schema={
        "market_id": pl.Utf8, "n": pl.UInt32,
        "p50_ms": pl.Float64, "p90_ms": pl.Float64, "p99_ms": pl.Float64,
        "mean_ms": pl.Float64, "max_ms": pl.Float64,
    })
