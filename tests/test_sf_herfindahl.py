import polars as pl

from polydata.sf.herfindahl import hhi_per_market


def test_hhi_pure_monopoly_is_one():
    df = pl.DataFrame({
        "market_id": ["m", "m", "m"],
        "maker": ["A", "A", "A"],
        "size": [1.0, 2.0, 3.0],
    })
    out = hhi_per_market(df)
    assert abs(out.row(0, named=True)["hhi"] - 1.0) < 1e-9


def test_hhi_equal_split_two_makers_is_half():
    df = pl.DataFrame({
        "market_id": ["m"] * 4,
        "maker": ["A", "A", "B", "B"],
        "size": [1.0, 1.0, 1.0, 1.0],
    })
    out = hhi_per_market(df)
    assert abs(out.row(0, named=True)["hhi"] - 0.5) < 1e-9


def test_hhi_empty_returns_empty():
    df = pl.DataFrame(
        schema={"market_id": pl.Utf8, "maker": pl.Utf8, "size": pl.Float64},
    )
    out = hhi_per_market(df)
    assert out.height == 0
