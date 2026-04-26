"""L2 depth profiles: top-k resting liquidity on each side.

The Polymarket ladder is sparse and non-uniform — prices do not lie on a
geometric grid. Top-k depth summed across populated levels is the natural
summary statistic.

LOBSample stores pre-computed depth for arbitrary k via depth_by_k. This
module reads that field directly for all k values.
"""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from polydata.window import MeasurementWindow

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


def _bid_depth(sample, k: int) -> float:
    bid_q, _ = sample.depth_by_k.get(k, (0, 0))
    return float(bid_q)


def _ask_depth(sample, k: int) -> float:
    _, ask_q = sample.depth_by_k.get(k, (0, 0))
    return float(ask_q)


def depth_series(window: MeasurementWindow, ks: Sequence[int] = DEFAULT_KS) -> pl.DataFrame:
    ts = [s.t for s in window.samples]
    cols: dict[str, list[float]] = {}
    for k in ks:
        cols[f"bid_top{k}"] = []
        cols[f"ask_top{k}"] = []
    for s in window.samples:
        for k in ks:
            cols[f"bid_top{k}"].append(_bid_depth(s, k))
            cols[f"ask_top{k}"].append(_ask_depth(s, k))
    data: dict[str, list] = {
        "market_id": [window.market_id] * len(ts),
        "t": ts,
    }
    data.update(cols)
    return pl.DataFrame(data)


def mean_depth_by_level(window: MeasurementWindow, ks: Sequence[int] = DEFAULT_KS) -> pl.DataFrame:
    series = depth_series(window, ks=ks)
    rows = []
    for k in ks:
        bid_col = f"bid_top{k}"
        ask_col = f"ask_top{k}"
        rows.append({
            "market_id": window.market_id,
            "k": int(k),
            "mean_bid_depth": float(series[bid_col].mean() or 0.0),
            "mean_ask_depth": float(series[ask_col].mean() or 0.0),
        })
    return pl.DataFrame(rows, schema={
        "market_id": pl.Utf8,
        "k": pl.UInt32,
        "mean_bid_depth": pl.Float64,
        "mean_ask_depth": pl.Float64,
    })
