"""YES + NO ≈ 1 parity check — no-arbitrage validation.

For any Polymarket binary market:
    mid_YES(t) + mid_NO(t) ≈ 1 - fee_spread

Deviations indicate (a) data-quality issues, (b) cross-side arbitrage
opportunities, or (c) the inferred-trade pipeline has the wrong token
direction. Report mean, max, and distribution of parity deviations.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from polydata.events import BookSnapshot, PriceChange
from polydata.window import MeasurementWindow


def yes_no_parity_check(
    window: MeasurementWindow,
    yes_token: str,
    no_token: str,
) -> pl.DataFrame:
    # Track latest best_bid/best_ask per token over time
    yes_bb: dict = {}
    yes_ba: dict = {}
    no_bb: dict = {}
    no_ba: dict = {}
    for rec in window.events:
        ev = rec.event
        if isinstance(ev, BookSnapshot | PriceChange):
            if ev.token_id == yes_token:
                if ev.best_bid is not None:
                    yes_bb[rec.ts_received] = float(ev.best_bid)
                if ev.best_ask is not None:
                    yes_ba[rec.ts_received] = float(ev.best_ask)
            elif ev.token_id == no_token:
                if ev.best_bid is not None:
                    no_bb[rec.ts_received] = float(ev.best_bid)
                if ev.best_ask is not None:
                    no_ba[rec.ts_received] = float(ev.best_ask)

    sums: list[float] = []
    last_yb = last_ya = last_nb = last_na = None
    for s in window.samples:
        for ts, v in yes_bb.items():
            if ts <= s.t:
                last_yb = v
        for ts, v in yes_ba.items():
            if ts <= s.t:
                last_ya = v
        for ts, v in no_bb.items():
            if ts <= s.t:
                last_nb = v
        for ts, v in no_ba.items():
            if ts <= s.t:
                last_na = v
        if None not in (last_yb, last_ya, last_nb, last_na):
            yes_mid = (last_yb + last_ya) / 2.0
            no_mid = (last_nb + last_na) / 2.0
            sums.append(yes_mid + no_mid)

    schema = {
        "market_id": pl.Utf8, "n_samples": pl.UInt32,
        "mean_parity_sum": pl.Float64,
        "max_abs_deviation": pl.Float64,
        "p99_abs_deviation": pl.Float64,
    }
    if not sums:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_samples": 0,
            "mean_parity_sum": float("nan"),
            "max_abs_deviation": float("nan"),
            "p99_abs_deviation": float("nan"),
        }], schema=schema)
    arr = np.asarray(sums)
    devs = np.abs(arr - 1.0)
    return pl.DataFrame([{
        "market_id": window.market_id, "n_samples": int(arr.size),
        "mean_parity_sum": float(arr.mean()),
        "max_abs_deviation": float(devs.max()),
        "p99_abs_deviation": float(np.percentile(devs, 99)),
    }], schema=schema)
