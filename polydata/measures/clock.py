"""Polygon-block alignment coefficient.

Polygon has ~2-second block cadence. Hypothesis (SF3): quote updates cluster
around block boundaries. We anchor the block grid at the window's t_start
(no absolute chain alignment is assumed — the RELATIVE clustering vs. a
control period is what matters). `aligned_share` is the fraction of events
whose ts_received falls within ± tolerance_ms of a multiple of
block_period_ms relative to t_start.
"""
from __future__ import annotations

import polars as pl

from polydata.events import PriceChange
from polydata.window import MeasurementWindow


def block_alignment(
    window: MeasurementWindow,
    block_period_ms: int = 2000,
    tolerance_ms: int = 100,
) -> pl.DataFrame:
    anchor = window.t_start
    aligned = 0
    total = 0
    for r in window.events:
        if not isinstance(r.event, PriceChange):
            continue
        total += 1
        dt_ms = (r.ts_received - anchor).total_seconds() * 1000.0
        phase = dt_ms % block_period_ms
        distance = min(phase, block_period_ms - phase)
        if distance <= tolerance_ms:
            aligned += 1
    share = (aligned / total) if total else 0.0
    return pl.DataFrame([{
        "market_id": window.market_id,
        "n": int(total),
        "aligned": int(aligned),
        "aligned_share": float(share),
        "block_period_ms": int(block_period_ms),
        "tolerance_ms": int(tolerance_ms),
    }], schema={
        "market_id": pl.Utf8,
        "n": pl.UInt32,
        "aligned": pl.UInt32,
        "aligned_share": pl.Float64,
        "block_period_ms": pl.UInt32,
        "tolerance_ms": pl.UInt32,
    })
