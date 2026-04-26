import polars as pl

from polydata.sf.latency import summarize_latency


def test_summarize_latency_gives_panel_medians():
    rows = [
        {"market_id": "m", "p50_ms": 100.0, "p90_ms": 500.0, "p99_ms": 2000.0,
         "n": 1000},
        {"market_id": "n", "p50_ms": 200.0, "p90_ms": 800.0, "p99_ms": 3000.0,
         "n": 500},
    ]
    df = pl.DataFrame(rows)
    out = summarize_latency(df)
    assert out.height == 1
    row = out.row(0, named=True)
    assert abs(row["median_p50"] - 150.0) < 1e-9
    assert abs(row["median_p90"] - 650.0) < 1e-9
    assert abs(row["median_p99"] - 2500.0) < 1e-9
    assert row["n_markets"] == 2


def test_summarize_latency_empty():
    df = pl.DataFrame(
        schema={
            "market_id": pl.Utf8, "n": pl.UInt32,
            "p50_ms": pl.Float64, "p90_ms": pl.Float64, "p99_ms": pl.Float64,
        },
    )
    out = summarize_latency(df)
    assert out.row(0, named=True)["n_markets"] == 0
