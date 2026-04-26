import polars as pl

from polydata.sf.wash import flag_self_counterparty, wash_share_per_market


def test_flags_same_maker_and_taker():
    df = pl.DataFrame({
        "market_id": ["m", "m", "m"],
        "block_number": [100, 101, 200],
        "maker": ["A", "B", "A"],
        "taker": ["A", "C", "B"],
        "size": [1.0, 1.0, 1.0],
    })
    out = flag_self_counterparty(df, window_blocks=128)
    flags = out.sort("block_number")["is_wash"].to_list()
    assert flags == [True, False, False]


def test_flags_roundtrip_within_window():
    df = pl.DataFrame({
        "market_id": ["m", "m"],
        "block_number": [100, 150],
        "maker": ["A", "B"],
        "taker": ["B", "A"],
        "size": [1.0, 1.0],
    })
    out = flag_self_counterparty(df, window_blocks=128)
    assert all(out["is_wash"].to_list())


def test_does_not_flag_outside_window():
    df = pl.DataFrame({
        "market_id": ["m", "m"],
        "block_number": [100, 500],
        "maker": ["A", "B"],
        "taker": ["B", "A"],
        "size": [1.0, 1.0],
    })
    out = flag_self_counterparty(df, window_blocks=128)
    assert not any(out["is_wash"].to_list())


def test_wash_share_per_market_aggregates():
    df = pl.DataFrame({
        "market_id": ["a", "a", "a", "a", "b", "b"],
        "is_wash": [True, True, False, False, False, False],
    })
    out = wash_share_per_market(df).sort("market_id")
    rows = out.to_dicts()
    assert rows[0]["market_id"] == "a" and abs(rows[0]["wash_share"] - 0.5) < 1e-9
    assert rows[1]["market_id"] == "b" and rows[1]["wash_share"] == 0.0
