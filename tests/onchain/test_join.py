import polars as pl

from polydata.onchain.join import (
    JoinResult,
    aggregate_onchain_by_block_tx,
    block_level_match,
    compute_metrics,
    decode_fill_to_trade,
)


def _fill(block, maker_aid, taker_aid, m_amt, t_amt, tx):
    return {
        "block_number": block, "tx_hash": tx, "log_index": 0,
        "order_hash": "0x" + "aa" * 32, "maker": "0x" + "bb" * 20,
        "taker": "0x" + "cc" * 20,
        "maker_asset_id": str(maker_aid), "taker_asset_id": str(taker_aid),
        "maker_amount_filled": str(m_amt), "taker_amount_filled": str(t_amt),
        "fee": "0",
    }


def test_decode_fill_buyer_aggressor():
    fill = _fill(1, maker_aid=12345, taker_aid=0,
                 m_amt=1000 * 10**6, t_amt=600 * 10**6, tx="0x" + "1" * 64)
    t = decode_fill_to_trade(fill)
    assert t["token_id"] == "12345"
    assert abs(t["price"] - 0.6) < 1e-9
    assert abs(t["size"] - 1000.0) < 1e-9
    assert t["sign"] == 1


def test_decode_fill_seller_aggressor():
    fill = _fill(1, maker_aid=0, taker_aid=98765,
                 m_amt=600 * 10**6, t_amt=1000 * 10**6, tx="0x" + "1" * 64)
    t = decode_fill_to_trade(fill)
    assert t["token_id"] == "98765"
    assert abs(t["price"] - 0.6) < 1e-9
    assert t["sign"] == -1


def test_aggregate_onchain_by_block_tx_sums_size_and_keeps_vw_price():
    fills = [
        _fill(10, 12345, 0, 1000 * 10**6, 600 * 10**6, "0x" + "1" * 64),
        _fill(10, 12345, 0, 500 * 10**6, 300 * 10**6, "0x" + "1" * 64),
    ]
    df = aggregate_onchain_by_block_tx(fills)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["block_number"] == 10
    assert row["tx_hash"] == "0x" + "1" * 64
    assert row["token_id"] == "12345"
    assert abs(row["price"] - 0.6) < 1e-9
    assert abs(row["size"] - 1500.0) < 1e-9


def test_block_level_match_tp_fp_fn():
    inferred = pl.DataFrame([
        {"market_id": "m", "block_number": 10, "token_id": "t1",
         "price": 0.6, "size": 1500.0, "sign": 1},
        {"market_id": "m", "block_number": 11, "token_id": "t1",
         "price": 0.7, "size": 500.0, "sign": 1},
    ], schema={
        "market_id": pl.Utf8, "block_number": pl.UInt64,
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    })
    onchain = pl.DataFrame([
        {"block_number": 10, "tx_hash": "0x1", "token_id": "t1",
         "price": 0.6, "size": 800.0, "sign": 1},
        {"block_number": 10, "tx_hash": "0x2", "token_id": "t1",
         "price": 0.6, "size": 700.0, "sign": 1},
        {"block_number": 12, "tx_hash": "0x3", "token_id": "t1",
         "price": 0.8, "size": 100.0, "sign": 1},
    ], schema={
        "block_number": pl.UInt64, "tx_hash": pl.Utf8,
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    })
    result = block_level_match(inferred, onchain)
    assert result.n_true_positive == 1
    assert result.n_false_positive == 1
    assert result.n_false_negative == 1
    assert result.size_mae == 0.0


def test_metrics_f1_perfect():
    r = JoinResult(n_inferred=10, n_onchain=10,
                   n_true_positive=10, n_false_positive=0, n_false_negative=0,
                   size_mae=0.0, sign_agreement=1.0)
    m = compute_metrics(r)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
