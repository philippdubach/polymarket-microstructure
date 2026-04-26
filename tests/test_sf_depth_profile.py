import polars as pl

from polydata.sf.depth_profile import depth_concentration


def test_concentration_ratio_per_market():
    rows = [
        # market m: top-of-book has 100, all 10 levels combined have 1000
        {"market_id": "m", "k": 1, "mean_bid_depth": 100.0,
         "mean_ask_depth": 100.0},
        {"market_id": "m", "k": 5, "mean_bid_depth": 500.0,
         "mean_ask_depth": 500.0},
        {"market_id": "m", "k": 10, "mean_bid_depth": 1000.0,
         "mean_ask_depth": 1000.0},
        # market n: book is concentrated at top (L1 = L10 essentially)
        {"market_id": "n", "k": 1, "mean_bid_depth": 95.0,
         "mean_ask_depth": 95.0},
        {"market_id": "n", "k": 5, "mean_bid_depth": 100.0,
         "mean_ask_depth": 100.0},
        {"market_id": "n", "k": 10, "mean_bid_depth": 100.0,
         "mean_ask_depth": 100.0},
    ]
    df = pl.DataFrame(rows)
    out = depth_concentration(df)
    m = out.filter(pl.col("market_id") == "m").row(0, named=True)
    n = out.filter(pl.col("market_id") == "n").row(0, named=True)
    # m: spread book — L1/L10 = 0.1; n: concentrated — L1/L10 = 0.95
    assert abs(m["concentration_l1_over_l10"] - 0.1) < 1e-9
    assert abs(n["concentration_l1_over_l10"] - 0.95) < 1e-9


def test_empty_input_returns_empty():
    df = pl.DataFrame(
        schema={
            "market_id": pl.Utf8, "k": pl.UInt32,
            "mean_bid_depth": pl.Float64, "mean_ask_depth": pl.Float64,
        },
    )
    out = depth_concentration(df)
    assert out.height == 0
