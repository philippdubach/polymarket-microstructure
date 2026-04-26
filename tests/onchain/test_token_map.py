from polydata.onchain.token_map import (
    TOKEN_MAP_SCHEMA,
    parse_gamma_clob_token_ids,
)


def test_parse_gamma_clob_token_ids_extracts_yes_no():
    row = {
        "conditionId": "0xabc",
        "clobTokenIds": '["11111", "22222"]',
    }
    yes, no = parse_gamma_clob_token_ids(row)
    assert yes == "11111"
    assert no == "22222"


def test_parse_gamma_clob_token_ids_missing_returns_none():
    row = {"conditionId": "0xabc"}
    yes, no = parse_gamma_clob_token_ids(row)
    assert yes is None
    assert no is None


def test_schema_columns():
    import polars as pl

    assert TOKEN_MAP_SCHEMA["market_id"] == pl.Utf8
    assert TOKEN_MAP_SCHEMA["yes_token_id"] == pl.Utf8
    assert TOKEN_MAP_SCHEMA["no_token_id"] == pl.Utf8
