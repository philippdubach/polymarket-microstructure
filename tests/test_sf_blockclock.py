import polars as pl

from polydata.sf.blockclock import alignment_summary


def test_alignment_counts_above_null():
    rows = [
        {"market_id": "m", "aligned_share": 0.10},
        {"market_id": "n", "aligned_share": 0.18},
        {"market_id": "o", "aligned_share": 0.30},
        {"market_id": "p", "aligned_share": 0.05},
    ]
    df = pl.DataFrame(rows)
    out = alignment_summary(df, null_share=0.10, threshold_above=0.15)
    row = out.row(0, named=True)
    assert row["n_markets"] == 4
    assert row["n_above_threshold"] == 2  # 0.18 and 0.30
    assert abs(row["share_above_threshold"] - 0.5) < 1e-9
    assert abs(row["median_alignment"] - 0.14) < 1e-9


def test_alignment_summary_empty():
    df = pl.DataFrame(schema={"market_id": pl.Utf8, "aligned_share": pl.Float64})
    out = alignment_summary(df)
    assert out.height == 1
    assert out.row(0, named=True)["n_markets"] == 0
