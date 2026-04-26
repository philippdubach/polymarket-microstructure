# polydata/measures/effective.py
"""Effective and realized spread with expert-mandated specifications.

- Pre-trade mid: at or before the (block-truncated) trade timestamp.
- Denominators: both `mid` and `min(p, 1-p)` are reported.
- Aggregation: dollar-volume weighted (primary) and equal-weighted.
- Realized spread: lag sweep over multiple horizons.
- Price impact: effective − realized per lag.
- Minimum trades: default 30; below threshold returns NaN with flag.

Uses STRICT inferred trades.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from polydata.measures._trade_source import resolve_trades
from polydata.window import MeasurementWindow

DEFAULT_LAGS_SEC: tuple[int, ...] = (15, 30, 60, 300)
DEFAULT_MIN_TRADES: int = 30
_NAN = float("nan")


def _mid_pre_trade(samples, ts) -> float | None:
    """Return the last mid at or before ts (inclusive).

    Trades are block-truncated so their timestamp equals the block boundary.
    The snapshot establishing the pre-trade state sits at the same boundary,
    but it was processed first, so using <= is correct.
    """
    prev_mid = None
    for s in samples:
        if s.t > ts:
            break
        if s.best_bid is not None and s.best_ask is not None:
            prev_mid = (float(s.best_bid) + float(s.best_ask)) / 2
    return prev_mid


def _mid_at_or_before(samples, ts) -> float | None:
    """Return the last mid at or before ts (inclusive)."""
    prev_mid = None
    for s in samples:
        if s.t > ts:
            break
        if s.best_bid is not None and s.best_ask is not None:
            prev_mid = (float(s.best_bid) + float(s.best_ask)) / 2
    return prev_mid


_EFF_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "n_trades": pl.UInt32,
    "dollar_volume": pl.Float64,
    "half_spread_prob_ew": pl.Float64,
    "half_spread_prob_dw": pl.Float64,
    "eff_spread_bps_mid_ew": pl.Float64,
    "eff_spread_bps_mid_dw": pl.Float64,
    "eff_spread_bps_minp_ew": pl.Float64,
    "eff_spread_bps_minp_dw": pl.Float64,
    "insufficient_trades_flag": pl.Boolean,
}


def _empty_eff_row(market_id: str, n: int, flag: bool) -> dict:
    return {
        "market_id": market_id,
        "n_trades": n,
        "dollar_volume": 0.0,
        "half_spread_prob_ew": _NAN,
        "half_spread_prob_dw": _NAN,
        "eff_spread_bps_mid_ew": _NAN,
        "eff_spread_bps_mid_dw": _NAN,
        "eff_spread_bps_minp_ew": _NAN,
        "eff_spread_bps_minp_dw": _NAN,
        "insufficient_trades_flag": flag,
    }


def effective_spread(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> pl.DataFrame:
    """Compute effective spread for a measurement window.

    Parameters
    ----------
    window:
        MeasurementWindow containing events and resampled LOB samples.
    trades:
        Optional trade DataFrame matching the canonical schema
        (market_id, t, token_id, price, size, sign). If None, falls back
        to STRICT inference from the window (deprecated per Plan 3b —
        orderbook-only inference has a 57% sign-agreement problem on
        Polymarket; pass authoritative on-chain trades instead).
    min_trades:
        Minimum number of trades required; below this the result row has
        NaN spread values and ``insufficient_trades_flag=True``.

    Returns
    -------
    pl.DataFrame
        One row per window with columns defined in ``_EFF_SCHEMA``.
    """
    trades = resolve_trades(window, trades)
    if trades.height == 0:
        return pl.DataFrame([_empty_eff_row(window.market_id, 0, True)], schema=_EFF_SCHEMA)

    halves: list[float] = []
    prices: list[float] = []
    sizes: list[float] = []
    mids: list[float] = []
    for row in trades.iter_rows(named=True):
        mid = _mid_pre_trade(window.samples, row["t"])
        if mid is None or mid <= 0 or mid >= 1:
            continue
        halves.append(row["sign"] * (row["price"] - mid))
        prices.append(row["price"])
        sizes.append(row["size"])
        mids.append(mid)

    n = len(halves)
    insufficient = n < min_trades
    if n == 0:
        return pl.DataFrame([_empty_eff_row(window.market_id, 0, True)], schema=_EFF_SCHEMA)
    if insufficient:
        return pl.DataFrame([_empty_eff_row(window.market_id, n, True)], schema=_EFF_SCHEMA)

    dollar_values = [p * s for p, s in zip(prices, sizes, strict=True)]
    total_dv = sum(dollar_values) or 1e-12

    hs_ew = sum(halves) / n
    hs_dw = sum(h * d for h, d in zip(halves, dollar_values, strict=True)) / total_dv
    bps_mid_ew = (
        sum(2 * abs(h) / m for h, m in zip(halves, mids, strict=True)) * 10_000.0 / n
    )
    bps_mid_dw = (
        sum(
            2 * abs(h) / m * d
            for h, m, d in zip(halves, mids, dollar_values, strict=True)
        )
        * 10_000.0 / total_dv
    )
    bps_minp_ew = (
        sum(
            2 * abs(h) / min(m, 1 - m)
            for h, m in zip(halves, mids, strict=True)
        )
        * 10_000.0 / n
    )
    bps_minp_dw = (
        sum(
            2 * abs(h) / min(m, 1 - m) * d
            for h, m, d in zip(halves, mids, dollar_values, strict=True)
        )
        * 10_000.0 / total_dv
    )

    return pl.DataFrame(
        [{
            "market_id": window.market_id,
            "n_trades": n,
            "dollar_volume": total_dv,
            "half_spread_prob_ew": hs_ew,
            "half_spread_prob_dw": hs_dw,
            "eff_spread_bps_mid_ew": bps_mid_ew,
            "eff_spread_bps_mid_dw": bps_mid_dw,
            "eff_spread_bps_minp_ew": bps_minp_ew,
            "eff_spread_bps_minp_dw": bps_minp_dw,
            "insufficient_trades_flag": False,
        }],
        schema=_EFF_SCHEMA,
    )


_REALIZED_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "lag_sec": pl.Int64,
    "n_trades": pl.UInt32,
    "realized_half_spread_ew": pl.Float64,
    "realized_half_spread_dw": pl.Float64,
    "price_impact_prob_ew": pl.Float64,
    "price_impact_prob_dw": pl.Float64,
    "insufficient_trades_flag": pl.Boolean,
}


def realized_spread(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
    lags_sec: list[int] | None = None,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> pl.DataFrame:
    """Compute realized spread and price impact for a sweep of lags.

    For each lag, the realized half-spread is:
        sign * (trade_price - mid_{t + lag})

    Price impact = effective half-spread - realized half-spread.

    Parameters
    ----------
    window:
        MeasurementWindow containing events and resampled LOB samples.
    trades:
        Optional trade DataFrame matching the canonical schema. If None,
        falls back to STRICT inference (see `effective_spread` docstring).
    lags_sec:
        List of lag horizons in seconds.
    min_trades:
        Minimum number of trades required for a valid estimate.

    Returns
    -------
    pl.DataFrame
        One row per lag with columns defined in ``_REALIZED_SCHEMA``.
    """
    if lags_sec is None:
        lags_sec = list(DEFAULT_LAGS_SEC)
    trades = resolve_trades(window, trades)
    rows: list[dict] = []

    def _empty_lag_row(lag: int, n: int) -> dict:
        return {
            "market_id": window.market_id, "lag_sec": lag, "n_trades": n,
            "realized_half_spread_ew": _NAN, "realized_half_spread_dw": _NAN,
            "price_impact_prob_ew": _NAN, "price_impact_prob_dw": _NAN,
            "insufficient_trades_flag": True,
        }

    if trades.height == 0:
        for lag in lags_sec:
            rows.append(_empty_lag_row(lag, 0))
        return pl.DataFrame(rows, schema=_REALIZED_SCHEMA)

    # Gather per-trade quantities for effective half-spread and dollar weight
    eff_halves: list[float] = []
    eff_dollar: list[float] = []
    ts_list: list = []
    signs: list[int] = []
    sizes: list[float] = []
    prices: list[float] = []

    for row in trades.iter_rows(named=True):
        mid_pre = _mid_pre_trade(window.samples, row["t"])
        if mid_pre is None or mid_pre <= 0 or mid_pre >= 1:
            continue
        eff_halves.append(row["sign"] * (row["price"] - mid_pre))
        eff_dollar.append(row["price"] * row["size"])
        ts_list.append(row["t"])
        signs.append(row["sign"])
        sizes.append(row["size"])
        prices.append(row["price"])

    n = len(ts_list)
    if n == 0:
        for lag in lags_sec:
            rows.append(_empty_lag_row(lag, 0))
        return pl.DataFrame(rows, schema=_REALIZED_SCHEMA)

    total_dv = sum(eff_dollar) or 1e-12
    insufficient = n < min_trades

    for lag in lags_sec:
        if insufficient:
            rows.append(_empty_lag_row(lag, n))
            continue
        rhs_values: list[float] = []
        rhs_dollars: list[float] = []
        for ts, sign, price, size in zip(ts_list, signs, prices, sizes, strict=True):
            mid_future = _mid_at_or_before(
                window.samples, ts + timedelta(seconds=lag)
            )
            if mid_future is None or mid_future <= 0 or mid_future >= 1:
                continue
            rhs_values.append(sign * (price - mid_future))
            rhs_dollars.append(price * size)
        if not rhs_values:
            rows.append(_empty_lag_row(lag, n))
            continue
        rhs_total_dv = sum(rhs_dollars) or 1e-12
        rhs_ew = sum(rhs_values) / len(rhs_values)
        rhs_dw = (
            sum(v * d for v, d in zip(rhs_values, rhs_dollars, strict=True))
            / rhs_total_dv
        )
        eff_ew = sum(eff_halves) / n
        eff_dw = (
            sum(v * d for v, d in zip(eff_halves, eff_dollar, strict=True)) / total_dv
        )
        rows.append({
            "market_id": window.market_id, "lag_sec": lag, "n_trades": n,
            "realized_half_spread_ew": rhs_ew, "realized_half_spread_dw": rhs_dw,
            "price_impact_prob_ew": eff_ew - rhs_ew,
            "price_impact_prob_dw": eff_dw - rhs_dw,
            "insufficient_trades_flag": False,
        })

    return pl.DataFrame(rows, schema=_REALIZED_SCHEMA)
