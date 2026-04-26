import polars as pl

from polydata.sf.category import aggregate_by_category, classify_question


def test_classify_keywords():
    assert classify_question("Will Bitcoin hit $100k?") == "Crypto"
    assert classify_question("Trump 2024 election winner") == "Politics"
    assert classify_question("Lakers vs Warriors winner") == "Sports"
    assert classify_question("Will Beyonce win Grammy?") == "Entertainment"
    assert classify_question("Apple Q3 earnings beat?") == "Business"
    assert classify_question("Random foo bar baz?") == "Other"


def test_aggregate_by_category_medians():
    rows = [
        {"market_id": "a", "category": "Sports",
         "value": 0.01},
        {"market_id": "b", "category": "Sports",
         "value": 0.03},
        {"market_id": "c", "category": "Politics",
         "value": 0.02},
    ]
    df = pl.DataFrame(rows)
    out = aggregate_by_category(df, "value")
    sp = out.filter(pl.col("category") == "Sports").row(0, named=True)
    po = out.filter(pl.col("category") == "Politics").row(0, named=True)
    assert abs(sp["median"] - 0.02) < 1e-9
    assert abs(po["median"] - 0.02) < 1e-9
    assert sp["n_markets"] == 2
    assert po["n_markets"] == 1
