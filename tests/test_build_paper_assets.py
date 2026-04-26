import polars as pl

from polydata.paper_assets import df_to_booktabs


def test_df_to_booktabs_renders_header_and_rows():
    df = pl.DataFrame({
        "category": ["Crypto", "Sports"],
        "median": [-0.039, 0.007],
        "n": [348, 142],
    })
    out = df_to_booktabs(df, header_map={
        "category": "Category",
        "median": "Median half-spread",
        "n": "Markets",
    })
    assert "\\begin{tabular}" in out
    assert "Category" in out
    assert "Crypto" in out
    assert "0.0070" in out
    assert "\\bottomrule" in out


def test_df_to_booktabs_handles_empty():
    df = pl.DataFrame(
        schema={"category": pl.Utf8, "median": pl.Float64, "n": pl.UInt32},
    )
    out = df_to_booktabs(df, header_map={
        "category": "Category", "median": "Median", "n": "N",
    })
    assert "\\begin{tabular}" in out
    assert "(no rows)" in out
