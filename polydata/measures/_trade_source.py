"""Shared helper: resolve an injected trades DataFrame or fall back to STRICT.

Plan 3b adds `trades: pl.DataFrame | None = None` to each trade-based
measure's signature. This helper centralizes the resolve-or-default logic
so the per-measure patches are one-liners.

Schema contract (both STRICT and authoritative on-chain producers conform):
    market_id: Utf8, t: Datetime(UTC), token_id: Utf8,
    price: Float64, size: Float64, sign: Int8
"""
from __future__ import annotations

import polars as pl

from polydata.measures.trades import infer_trades_strict
from polydata.window import MeasurementWindow


def resolve_trades(
    window: MeasurementWindow,
    trades: pl.DataFrame | None,
) -> pl.DataFrame:
    if trades is not None:
        return trades
    return infer_trades_strict(window)
