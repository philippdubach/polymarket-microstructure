"""SF8: depth decay near resolution.

Two flavours:

A. Per-market within-window regression: depth_t ~ log(seconds_to_close).
   Requires a per-market time-series of depth — T4 currently emits
   summaries only, so this is deferred to a T4 re-run.

B. Cross-sectional panel regression: per-market summary depth vs.
   per-market average seconds_to_close at the panel window midpoint.
   Tests whether markets that are nearer to resolution during the
   window have shallower books on average. Implemented below using
   `data/panel_metadata.parquet` (CLOB REST `end_date_iso`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm


def fit_depth_decay(per_market_series: pl.DataFrame) -> pl.DataFrame:
    """Per-market OLS: depth ~ log(seconds_to_close) + intercept."""
    rows: list[dict] = []
    for mid, sub in per_market_series.group_by("market_id"):
        clean = sub.filter(
            (pl.col("seconds_to_close") > 0)
            & pl.col("mean_depth").is_not_null()
        )
        if clean.height < 10:
            continue
        x = np.log(clean["seconds_to_close"].to_numpy())
        y = clean["mean_depth"].to_numpy()
        x_const = sm.add_constant(x)
        res = sm.OLS(y, x_const).fit()
        rows.append({
            "market_id": mid[0] if isinstance(mid, tuple) else mid,
            "intercept": float(res.params[0]),
            "slope_log_ttc": float(res.params[1]),
            "r2": float(res.rsquared) if not np.isnan(res.rsquared) else float("nan"),
            "n_obs": int(clean.height),
        })
    return pl.DataFrame(rows)


def fit_with_category_fe(per_market: pl.DataFrame) -> dict:
    """Cross-sectional panel regression with category fixed effects:
    log(mean_depth) ~ log(seconds_to_close) + C(category).
    Returns slope on log_ttc with HC3-robust SE.
    """
    clean = per_market.filter(
        (pl.col("seconds_to_close") > 0)
        & (pl.col("mean_depth") > 0)
        & pl.col("category").is_not_null()
    )
    if clean.height < 30:
        return {
            "slope_log_ttc": float("nan"),
            "se": float("nan"), "r2": float("nan"),
            "n_obs": int(clean.height),
            "n_categories": 0,
        }
    import statsmodels.formula.api as smf
    df = clean.with_columns(
        pl.col("seconds_to_close").log().alias("log_ttc"),
        pl.col("mean_depth").log().alias("log_depth"),
    ).to_pandas()
    res = smf.ols(
        "log_depth ~ log_ttc + C(category)", data=df,
    ).fit(cov_type="HC3")
    slope = float(res.params.get("log_ttc", float("nan")))
    return {
        "slope_log_ttc": slope,
        "se": float(res.bse.get("log_ttc", float("nan"))),
        "r2": float(res.rsquared) if not np.isnan(res.rsquared) else float("nan"),
        "n_obs": int(clean.height),
        "n_categories": int(df["category"].nunique()),
    }


def fit_with_category_and_volume_fe(per_market: pl.DataFrame) -> dict:
    """Cross-sectional regression: log(mean_depth) ~ log(seconds_to_close)
    + log(volume_usd) + C(category). HC3 SE.
    Tests whether the depth-decay slope survives controlling for total
    on-chain volume in addition to category.
    """
    clean = per_market.filter(
        (pl.col("seconds_to_close") > 0)
        & (pl.col("mean_depth") > 0)
        & (pl.col("volume_usd") > 0)
        & pl.col("category").is_not_null()
    )
    if clean.height < 30:
        return {
            "slope_log_ttc": float("nan"),
            "se": float("nan"), "r2": float("nan"),
            "n_obs": int(clean.height),
            "n_categories": 0,
        }
    import statsmodels.formula.api as smf
    df = clean.with_columns(
        pl.col("seconds_to_close").log().alias("log_ttc"),
        pl.col("mean_depth").log().alias("log_depth"),
        pl.col("volume_usd").log().alias("log_volume"),
    ).to_pandas()
    res = smf.ols(
        "log_depth ~ log_ttc + log_volume + C(category)", data=df,
    ).fit(cov_type="HC3")
    slope = float(res.params.get("log_ttc", float("nan")))
    return {
        "slope_log_ttc": slope,
        "se": float(res.bse.get("log_ttc", float("nan"))),
        "r2": float(res.rsquared) if not np.isnan(res.rsquared) else float("nan"),
        "n_obs": int(clean.height),
        "n_categories": int(df["category"].nunique()),
        "slope_log_volume": float(res.params.get("log_volume", float("nan"))),
        "se_log_volume": float(res.bse.get("log_volume", float("nan"))),
    }


def fit_cross_sectional(per_market: pl.DataFrame) -> dict:
    """Cross-sectional panel regression: log(mean_depth) ~ log(seconds_to_close).

    Negative slope = markets nearer to resolution have shallower books.
    Returns slope, se, R², n_obs.
    """
    clean = per_market.filter(
        (pl.col("seconds_to_close") > 0)
        & (pl.col("mean_depth") > 0)
    )
    if clean.height < 10:
        return {
            "slope_log_ttc": float("nan"),
            "se": float("nan"), "r2": float("nan"),
            "n_obs": int(clean.height),
        }
    x = np.log(clean["seconds_to_close"].to_numpy())
    y = np.log(clean["mean_depth"].to_numpy())
    x_const = sm.add_constant(x)
    res = sm.OLS(y, x_const).fit(cov_type="HC3")
    return {
        "slope_log_ttc": float(res.params[1]),
        "se": float(res.bse[1]),
        "r2": float(res.rsquared) if not np.isnan(res.rsquared) else float("nan"),
        "intercept": float(res.params[0]),
        "n_obs": int(clean.height),
    }


def run_sf8(
    panel_path: Path = Path("data/panel.parquet"),
    metadata_path: Path = Path("data/panel_metadata.parquet"),
    depth_path: Path = Path("data/panel_quote/depth.parquet"),
    out_parquet: Path = Path("artifacts/sf8_depth_decay.parquet"),
    out_png: Path = Path("artifacts/sf8_depth_decay.png"),
    window_midpoint_iso: str = "2026-03-13T00:00:00+00:00",
) -> pl.DataFrame:
    """Cross-sectional panel regression of log(mean depth) on
    log(seconds_to_close at window midpoint), with and without
    category fixed effects."""
    from datetime import datetime

    import matplotlib.pyplot as plt

    from polydata.sf.category import classify_question

    panel = pl.read_parquet(panel_path)
    meta = pl.read_parquet(metadata_path).select(
        ["market_id", "end_date_iso", "question"]
    ).with_columns(
        pl.col("question").map_elements(
            classify_question, return_dtype=pl.Utf8,
        ).alias("category"),
    )
    depth = pl.read_parquet(depth_path).filter(pl.col("k") == 10).select([
        "market_id",
        ((pl.col("mean_bid_depth") + pl.col("mean_ask_depth")) / 2)
        .alias("mean_depth"),
    ])
    midpoint = datetime.fromisoformat(window_midpoint_iso)
    joined = (
        panel.select(["market_id", "volume_usd"])
        .join(meta, on="market_id", how="inner")
        .join(depth, on="market_id", how="inner")
        .with_columns(
            pl.col("end_date_iso").str.to_datetime(time_zone="UTC")
              .alias("end_dt"),
        )
        .with_columns(
            ((pl.col("end_dt") - pl.lit(midpoint)).dt.total_seconds())
            .alias("seconds_to_close"),
        )
        .select([
            "market_id", "mean_depth", "seconds_to_close", "category",
            "volume_usd",
        ])
    )
    fit = fit_cross_sectional(joined)
    fit_fe = fit_with_category_fe(joined)
    fit_fe_vol = fit_with_category_and_volume_fe(joined)
    out = pl.DataFrame([
        {
            "spec": "bivariate",
            "n_markets": int(joined.height),
            "n_used_in_fit": fit["n_obs"],
            "slope_log_ttc": fit["slope_log_ttc"],
            "intercept": fit.get("intercept", float("nan")),
            "se_hc3": fit["se"],
            "r2": fit["r2"],
            "n_categories": 0,
        },
        {
            "spec": "category_fe",
            "n_markets": int(joined.height),
            "n_used_in_fit": fit_fe["n_obs"],
            "slope_log_ttc": fit_fe["slope_log_ttc"],
            "intercept": float("nan"),
            "se_hc3": fit_fe["se"],
            "r2": fit_fe["r2"],
            "n_categories": fit_fe["n_categories"],
        },
        {
            "spec": "category_fe_volume",
            "n_markets": int(joined.height),
            "n_used_in_fit": fit_fe_vol["n_obs"],
            "slope_log_ttc": fit_fe_vol["slope_log_ttc"],
            "intercept": float("nan"),
            "se_hc3": fit_fe_vol["se"],
            "r2": fit_fe_vol["r2"],
            "n_categories": fit_fe_vol["n_categories"],
        },
    ])
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = joined.filter(
        (pl.col("seconds_to_close") > 0)
        & (pl.col("mean_depth") > 0)
    )
    if plot_df.height > 0:
        x = np.log(plot_df["seconds_to_close"].to_numpy())
        y = np.log(plot_df["mean_depth"].to_numpy())
        ax.scatter(x, y, alpha=0.4, s=10)
        if not np.isnan(fit["slope_log_ttc"]):
            xline = np.linspace(x.min(), x.max(), 50)
            yline = fit.get("intercept", 0) + fit["slope_log_ttc"] * xline
            ax.plot(xline, yline, "r-",
                    label=f"slope={fit['slope_log_ttc']:.3f} (R²={fit['r2']:.3f})")
        ax.legend()
    ax.set_xlabel("log(seconds to close at window midpoint)")
    ax.set_ylabel("log(mean depth at L10)")
    ax.set_title("SF8 — cross-sectional depth vs. time-to-close")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out
