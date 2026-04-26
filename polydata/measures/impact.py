# polydata/measures/impact.py
"""Kyle's λ and Amihud illiquidity on log-odds (expert-refined spec).

- Winsorize p at [0.001, 0.999] before log-odds transform.
- Kyle: OLS of Δy (log-odds) on signed_volume with intercept.
        Report point, HAC (Newey-West, 4 lags) SE, HC3 SE, R²,
        plus a price-space Δp robustness column.
- Amihud: ratio-of-means (primary, Lou-Shu 2017) + mean-of-ratios
  (secondary). Winsorize p; use dollar_volume = size * price.
- Bucket default: 1 minute (60 buckets/hour).
- Minimum 10 buckets for a reportable estimate.

Uses STRICT inferred trades.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import polars as pl
import statsmodels.api as sm

from polydata.measures._trade_source import resolve_trades
from polydata.window import MeasurementWindow

DEFAULT_BUCKET = timedelta(minutes=1)
DEFAULT_MIN_BUCKETS = 10
WINSOR_LO = 0.001
WINSOR_HI = 0.999
HAC_LAGS = 4


def _logodds(p: float) -> float:
    p = max(WINSOR_LO, min(WINSOR_HI, p))
    return math.log(p / (1.0 - p))


def _bucketize(trades: pl.DataFrame, bucket: timedelta) -> pl.DataFrame:
    if trades.height == 0:
        return trades
    return (
        trades
        .with_columns(pl.col("t").dt.truncate(bucket).alias("bucket_t"))
        .group_by("bucket_t")
        .agg(
            (pl.col("size") * pl.col("sign")).sum().alias("signed_volume"),
            (pl.col("size") * pl.col("price")).sum().alias("dollar_volume"),
        )
        .sort("bucket_t")
    )


def _bucket_last_mid(samples, bucket_t, bucket: timedelta) -> float | None:
    last_mid = None
    end = bucket_t + bucket
    for s in samples:
        if s.t < bucket_t:
            continue
        if s.t >= end:
            break
        if s.best_bid is not None and s.best_ask is not None:
            last_mid = (float(s.best_bid) + float(s.best_ask)) / 2
    return last_mid


_KYLE_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "n_buckets": pl.UInt32,
    "kyle_lambda_logodds": pl.Float64,
    "kyle_lambda_price": pl.Float64,
    "kyle_intercept": pl.Float64,
    "kyle_r2": pl.Float64,
    "kyle_se_hac": pl.Float64,
    "kyle_se_hc3": pl.Float64,
    "insufficient_buckets_flag": pl.Boolean,
}


def _kyle_empty(market_id: str, n: int) -> dict:
    return {
        "market_id": market_id,
        "n_buckets": n,
        "kyle_lambda_logodds": float("nan"),
        "kyle_lambda_price": float("nan"),
        "kyle_intercept": float("nan"),
        "kyle_r2": float("nan"),
        "kyle_se_hac": float("nan"),
        "kyle_se_hc3": float("nan"),
        "insufficient_buckets_flag": True,
    }


def kyle_lambda(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
    bucket: timedelta = DEFAULT_BUCKET,
    min_buckets: int = DEFAULT_MIN_BUCKETS,
) -> pl.DataFrame:
    trades = resolve_trades(window, trades)
    bk = _bucketize(trades, bucket)
    if bk.height < min_buckets:
        return pl.DataFrame([_kyle_empty(window.market_id, int(bk.height))], schema=_KYLE_SCHEMA)

    y_vals: list[float] = []
    dp_vals: list[float] = []
    x_vals: list[float] = []
    prev_y: float | None = None
    prev_mid: float | None = None
    for row in bk.iter_rows(named=True):
        mid = _bucket_last_mid(window.samples, row["bucket_t"], bucket)
        if mid is None:
            continue
        y = _logodds(mid)
        if prev_y is not None and prev_mid is not None:
            y_vals.append(y - prev_y)
            dp_vals.append(mid - prev_mid)
            x_vals.append(float(row["signed_volume"]))
        prev_y, prev_mid = y, mid

    if len(y_vals) < min_buckets:
        return pl.DataFrame([_kyle_empty(window.market_id, int(bk.height))], schema=_KYLE_SCHEMA)

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    dp = np.asarray(dp_vals, dtype=float)
    x_with_const = sm.add_constant(x, has_constant="add")
    ols = sm.OLS(y, x_with_const).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)
    hc3 = ols.get_robustcov_results(cov_type="HC3")
    ols_price = sm.OLS(dp, x_with_const).fit()

    slope_idx = 1
    return pl.DataFrame(
        [
            {
                "market_id": window.market_id,
                "n_buckets": int(len(y_vals)),
                "kyle_lambda_logodds": float(ols.params[slope_idx]),
                "kyle_lambda_price": float(ols_price.params[slope_idx]),
                "kyle_intercept": float(ols.params[0]),
                "kyle_r2": float(ols.rsquared)
                if not math.isnan(ols.rsquared)
                else float("nan"),
                "kyle_se_hac": float(hac.bse[slope_idx]),
                "kyle_se_hc3": float(hc3.bse[slope_idx]),
                "insufficient_buckets_flag": False,
            }
        ],
        schema=_KYLE_SCHEMA,
    )


_AMIHUD_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "n_buckets": pl.UInt32,
    "amihud_ratio_of_means": pl.Float64,
    "amihud_mean_of_ratios": pl.Float64,
    "insufficient_buckets_flag": pl.Boolean,
}


def amihud_illiquidity(
    window: MeasurementWindow,
    trades: pl.DataFrame | None = None,
    bucket: timedelta = DEFAULT_BUCKET,
    min_buckets: int = DEFAULT_MIN_BUCKETS,
) -> pl.DataFrame:
    trades = resolve_trades(window, trades)
    bk = _bucketize(trades, bucket)
    if bk.height < min_buckets:
        return pl.DataFrame(
            [
                {
                    "market_id": window.market_id,
                    "n_buckets": int(bk.height),
                    "amihud_ratio_of_means": float("nan"),
                    "amihud_mean_of_ratios": float("nan"),
                    "insufficient_buckets_flag": True,
                }
            ],
            schema=_AMIHUD_SCHEMA,
        )

    abs_dy: list[float] = []
    dv: list[float] = []
    ratios: list[float] = []
    prev_y: float | None = None
    for row in bk.iter_rows(named=True):
        mid = _bucket_last_mid(window.samples, row["bucket_t"], bucket)
        if mid is None:
            continue
        y = _logodds(mid)
        dv_bucket = float(row["dollar_volume"])
        if prev_y is not None and dv_bucket > 0:
            d = abs(y - prev_y)
            abs_dy.append(d)
            dv.append(dv_bucket)
            ratios.append(d / dv_bucket)
        prev_y = y

    if not ratios:
        return pl.DataFrame(
            [
                {
                    "market_id": window.market_id,
                    "n_buckets": int(bk.height),
                    "amihud_ratio_of_means": float("nan"),
                    "amihud_mean_of_ratios": float("nan"),
                    "insufficient_buckets_flag": True,
                }
            ],
            schema=_AMIHUD_SCHEMA,
        )

    rom = sum(abs_dy) / sum(dv) if sum(dv) > 0 else float("nan")
    mor = float(np.mean(ratios))
    return pl.DataFrame(
        [
            {
                "market_id": window.market_id,
                "n_buckets": int(bk.height),
                "amihud_ratio_of_means": rom,
                "amihud_mean_of_ratios": mor,
                "insufficient_buckets_flag": False,
            }
        ],
        schema=_AMIHUD_SCHEMA,
    )
