# polydata/measures/spread_est.py
"""Abdi-Ranaldo (2017) and Roll (1984) spread estimators on log-odds prices.

Expert recommendation (review 3): Roll NaN-rate 30-55% on typical windows;
cancel contamination biases Roll 1.3-2× upward. AR is a drop-in replacement
that handles the positive-covariance pathology.

Both estimators operate on STRICT inferred trades' log-odds prices. We
winsorize p at [0.001, 0.999] before the log-odds transform to keep the
transformed series finite.

- AR: c_t - η_t cross-products; S = 2 * sqrt(max(0, E[Δ_t Δ_{t+1}])).
  Report point + stationary-bootstrap 95% CI.
- Roll: spread = 2 * sqrt(-Cov(Δy_t, Δy_{t-1})) — NaN if Cov >= 0.

Minimum n_trades = 100; below threshold returns NaN with an insufficient
flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from polydata.measures._trade_source import resolve_trades
from polydata.window import MeasurementWindow

WINSOR_LO: float = 0.001
WINSOR_HI: float = 0.999
DEFAULT_MIN_TRADES: int = 100
N_BOOTSTRAP: int = 500


def _logodds(p: float) -> float:
    p = max(WINSOR_LO, min(WINSOR_HI, p))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class _Estimate:
    point: float
    ci_lo: float
    ci_hi: float
    n: int


def _ar_estimate_from_series(
    logodds_prices: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> _Estimate:
    """AR-style estimator on a log-odds price series.

    Treats the series as successive bucket closes; for each adjacent
    pair (c_t, c_{t+1}) the high is max(c_t, c_{t+1}) and low is min(...),
    so η_t = (high+low)/2 = (c_t + c_{t+1})/2 and c_t - η_t =
    (c_t - c_{t+1})/2. The cross-product E[(c_t-η_t)(c_t-η_{t+1})] captures
    the bid-ask bounce signature. Bootstrap the cross-products for a
    95% CI.
    """
    arr = np.asarray(logodds_prices, dtype=float)
    n = int(arr.size)
    if n < 3:
        return _Estimate(float("nan"), float("nan"), float("nan"), n)
    hi = np.maximum(arr[:-1], arr[1:])
    lo = np.minimum(arr[:-1], arr[1:])
    eta = (hi + lo) / 2.0
    c_inner = arr[:-1]
    delta = c_inner - eta
    if delta.size < 2:
        return _Estimate(float("nan"), float("nan"), float("nan"), n)

    def ar_point(d: np.ndarray) -> float:
        if d.size < 2:
            return 0.0
        cross = d[:-1] * d[1:]
        mean = float(cross.mean())
        if mean <= 0:
            return 0.0
        return 2.0 * math.sqrt(mean)

    point = ar_point(delta)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, delta.size, size=delta.size)
        samples.append(ar_point(delta[idx]))
    s_arr = np.asarray(samples)
    return _Estimate(
        point=point,
        ci_lo=float(np.percentile(s_arr, 2.5)),
        ci_hi=float(np.percentile(s_arr, 97.5)),
        n=n,
    )


def _trade_series_logodds(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
) -> list[float]:
    trades = resolve_trades(window, trades)
    if trades.height == 0:
        return []
    return [_logodds(float(p)) for p in trades["price"].to_list()]


_AR_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "n_trades": pl.UInt32,
    "ar_half_spread_logodds": pl.Float64,
    "ar_ci_lo_logodds": pl.Float64,
    "ar_ci_hi_logodds": pl.Float64,
    "insufficient_trades_flag": pl.Boolean,
}

_ROLL_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "n_trades": pl.UInt32,
    "roll_half_spread_logodds": pl.Float64,
    "roll_cov_positive_flag": pl.Boolean,
    "insufficient_trades_flag": pl.Boolean,
}


def abdi_ranaldo_spread(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> pl.DataFrame:
    y = _trade_series_logodds(window, trades)
    n = len(y)
    if n < min_trades:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_trades": n,
            "ar_half_spread_logodds": float("nan"),
            "ar_ci_lo_logodds": float("nan"),
            "ar_ci_hi_logodds": float("nan"),
            "insufficient_trades_flag": True,
        }], schema=_AR_SCHEMA)
    est = _ar_estimate_from_series(y)
    return pl.DataFrame([{
        "market_id": window.market_id, "n_trades": n,
        "ar_half_spread_logodds": est.point / 2.0,
        "ar_ci_lo_logodds": est.ci_lo / 2.0,
        "ar_ci_hi_logodds": est.ci_hi / 2.0,
        "insufficient_trades_flag": False,
    }], schema=_AR_SCHEMA)


def roll_implied_spread(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> pl.DataFrame:
    y = _trade_series_logodds(window, trades)
    n = len(y)
    if n < min_trades:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_trades": n,
            "roll_half_spread_logodds": float("nan"),
            "roll_cov_positive_flag": False,
            "insufficient_trades_flag": True,
        }], schema=_ROLL_SCHEMA)
    dy = np.diff(np.asarray(y))
    if dy.size < 2:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_trades": n,
            "roll_half_spread_logodds": float("nan"),
            "roll_cov_positive_flag": False,
            "insufficient_trades_flag": True,
        }], schema=_ROLL_SCHEMA)
    cov = float(np.cov(dy[:-1], dy[1:], bias=True)[0, 1])
    if cov >= 0:
        return pl.DataFrame([{
            "market_id": window.market_id, "n_trades": n,
            "roll_half_spread_logodds": float("nan"),
            "roll_cov_positive_flag": True,
            "insufficient_trades_flag": False,
        }], schema=_ROLL_SCHEMA)
    return pl.DataFrame([{
        "market_id": window.market_id, "n_trades": n,
        "roll_half_spread_logodds": math.sqrt(-cov),
        "roll_cov_positive_flag": False,
        "insufficient_trades_flag": False,
    }], schema=_ROLL_SCHEMA)
