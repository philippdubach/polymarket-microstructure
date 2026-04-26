import polars as pl

from polydata.panel.stratify import select_random, select_top_n


def _stats(rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"market_id": pl.Utf8, "n_trades": pl.UInt32,
                "volume_usd": pl.Float64},
        orient="row",
    )


def test_select_top_n_returns_sorted_descending():
    stats = _stats([("a", 10, 1000.0), ("b", 20, 2000.0), ("c", 30, 500.0)])
    top = select_top_n(stats, n=2)
    assert top["market_id"].to_list() == ["b", "a"]


def test_select_top_n_tiebreaks_by_n_trades_then_market_id():
    stats = _stats([
        ("x", 5, 100.0), ("y", 10, 100.0), ("a", 10, 100.0),
    ])
    top = select_top_n(stats, n=3)
    # same volume: higher n_trades wins; then lex ascending
    assert top["market_id"].to_list() == ["a", "y", "x"]


def test_select_random_excludes_top_and_respects_min_trades():
    stats = _stats([
        ("top1", 5000, 9999.0),
        ("q1", 200, 50.0),
        ("q2", 150, 50.0),
        ("skip_low_trades", 50, 50.0),
    ])
    top = select_top_n(stats, n=1)
    rand = select_random(stats, top, n=2, min_trades=100, seed=42)
    picked = set(rand["market_id"].to_list())
    assert "top1" not in picked
    assert "skip_low_trades" not in picked
    assert picked == {"q1", "q2"}


def test_select_random_returns_empty_when_universe_exhausted():
    stats = _stats([("top1", 5000, 100.0), ("q1", 200, 50.0)])
    top = select_top_n(stats, n=1)
    rand = select_random(stats, top, n=10, min_trades=100, seed=1)
    assert rand["market_id"].to_list() == ["q1"]
