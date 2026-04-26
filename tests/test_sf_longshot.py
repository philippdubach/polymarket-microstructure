import polars as pl

from polydata.sf.longshot import bin_spreads_by_price


def test_bin_spreads_monotone_center_lower_than_edges():
    rows = [
        {"market_id": f"m{i}", "mean_mid": m, "median_spread_bps": sp}
        for i, (m, sp) in enumerate([
            (0.03, 400), (0.08, 300),
            (0.45, 80), (0.50, 75),
            (0.92, 320), (0.97, 410),
        ])
    ]
    df = pl.DataFrame(rows)
    out = bin_spreads_by_price(df, n_bins=3)
    assert out["bin"].to_list() == [0, 1, 2]
    center = out.filter(pl.col("bin") == 1)["median_spread_bps"][0]
    edges = out.filter(pl.col("bin") != 1)["median_spread_bps"].to_list()
    assert center < min(edges)


def test_bin_spreads_handles_empty():
    df = pl.DataFrame(
        schema={"market_id": pl.Utf8, "mean_mid": pl.Float64,
                "median_spread_bps": pl.Float64},
    )
    out = bin_spreads_by_price(df, n_bins=10)
    assert out.height == 0
