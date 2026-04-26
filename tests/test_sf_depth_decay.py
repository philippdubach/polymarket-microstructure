import numpy as np
import polars as pl

from polydata.sf.depth_decay import fit_depth_decay


def test_depth_decay_recovers_negative_slope_when_depth_drops_with_log_ttc():
    rng = np.random.default_rng(0)
    ttc = np.linspace(100000, 1, 200)
    # depth grows with log(ttc) (so depth FALLS as resolution approaches)
    depth = 10000 - 200 * np.log(ttc) + rng.normal(0, 50, ttc.size)
    df = pl.DataFrame({
        "market_id": ["m"] * ttc.size,
        "seconds_to_close": ttc,
        "mean_depth": depth,
    })
    out = fit_depth_decay(df)
    assert out.height == 1
    # Higher ttc → higher depth here, so slope on log(ttc) is negative
    # (because we set depth = 10000 - 200 * log(ttc))
    slope = out.row(0, named=True)["slope_log_ttc"]
    assert slope < -100
