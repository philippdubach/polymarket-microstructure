import polars as pl

from polydata.sf.glosten_harris import decompose_per_market


def test_decompose_subtracts_realized_from_effective():
    rows = [
        {"market_id": "a", "eff_half": 0.005, "realized_half": 0.001},
        {"market_id": "b", "eff_half": 0.003, "realized_half": -0.001},
    ]
    df = pl.DataFrame(rows)
    out = decompose_per_market(df).sort("market_id")
    a = out.filter(pl.col("market_id") == "a").row(0, named=True)
    b = out.filter(pl.col("market_id") == "b").row(0, named=True)
    assert abs(a["c_transitory"] - 0.001) < 1e-12
    assert abs(a["phi_adverse_sel"] - 0.004) < 1e-12
    assert abs(b["c_transitory"] - (-0.001)) < 1e-12
    assert abs(b["phi_adverse_sel"] - 0.004) < 1e-12


def test_decompose_empty():
    df = pl.DataFrame(
        schema={
            "market_id": pl.Utf8, "eff_half": pl.Float64,
            "realized_half": pl.Float64,
        },
    )
    out = decompose_per_market(df)
    assert out.height == 0
