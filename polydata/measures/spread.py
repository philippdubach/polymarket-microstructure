"""Quoted-spread measures.

Half-spread s_q(t) = (a(t) - b(t)) / 2 where a, b are best ask / bid.
Price-conditional s_q(p) partitions observations by mid-price bin to measure
the longshot spread premium (SF1).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from polydata.window import MeasurementWindow


def quoted_spread_series(window: MeasurementWindow) -> pl.DataFrame:
    ts = [s.t for s in window.samples]
    mids: list[float | None] = []
    halves: list[float | None] = []
    for s in window.samples:
        if s.best_bid is None or s.best_ask is None:
            mids.append(None)
            halves.append(None)
        else:
            a = float(s.best_ask)
            b = float(s.best_bid)
            mids.append((a + b) / 2)
            halves.append((a - b) / 2)
    return pl.DataFrame({
        "market_id": [window.market_id] * len(ts),
        "t": ts,
        "mid": mids,
        "half_spread": halves,
    }, schema={
        "market_id": pl.Utf8,
        "t": pl.Datetime(time_zone="UTC"),
        "mid": pl.Float64,
        "half_spread": pl.Float64,
    })


def price_conditional_spread(
    window: MeasurementWindow,
    n_bins: int = 20,
) -> pl.DataFrame:
    series = quoted_spread_series(window)
    clean = series.drop_nulls(subset=["mid", "half_spread"])
    if clean.height == 0:
        return pl.DataFrame(schema={
            "market_id": pl.Utf8,
            "price_bin_lo": pl.Float64,
            "price_bin_hi": pl.Float64,
            "n": pl.UInt32,
            "mean_half_spread": pl.Float64,
        })
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    lo = edges[:-1]
    hi = edges[1:]
    mids = clean["mid"].to_numpy()
    hs = clean["half_spread"].to_numpy()
    rows = []
    for i in range(n_bins):
        last = i == n_bins - 1
        mask = (mids >= lo[i]) & (mids <= hi[i]) if last else (mids >= lo[i]) & (mids < hi[i])
        if not mask.any():
            continue
        rows.append({
            "market_id": window.market_id,
            "price_bin_lo": float(lo[i]),
            "price_bin_hi": float(hi[i]),
            "n": int(mask.sum()),
            "mean_half_spread": float(hs[mask].mean()),
        })
    return pl.DataFrame(rows, schema={
        "market_id": pl.Utf8,
        "price_bin_lo": pl.Float64,
        "price_bin_hi": pl.Float64,
        "n": pl.UInt32,
        "mean_half_spread": pl.Float64,
    })
