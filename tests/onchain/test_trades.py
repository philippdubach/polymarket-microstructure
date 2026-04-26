from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from polydata.onchain.trades import ONCHAIN_TRADE_SCHEMA, OnchainTrade


def test_onchain_trade_dataclass_fields():
    t = OnchainTrade(
        ts=datetime(2026, 3, 1, tzinfo=UTC),
        block_number=83_558_384,
        tx_hash="0x" + "de" * 32,
        token_id="123456789",
        price=Decimal("0.6"),
        size=Decimal("1000"),
        sign=1,
        maker="0x" + "aa" * 20,
        taker="0x" + "bb" * 20,
    )
    assert t.sign == 1
    assert t.price == Decimal("0.6")


def test_onchain_trade_schema_matches_measures_contract():
    # Plan 3b measures consume a DataFrame with these columns.
    required = {"market_id", "t", "token_id", "price", "size", "sign"}
    assert required.issubset(set(ONCHAIN_TRADE_SCHEMA.keys()))
    assert ONCHAIN_TRADE_SCHEMA["price"] == pl.Float64
    assert ONCHAIN_TRADE_SCHEMA["size"] == pl.Float64
    assert ONCHAIN_TRADE_SCHEMA["sign"] == pl.Int8


def _write_fake_slice(tmp_path, fills):
    import polars as pl
    df = pl.DataFrame(fills, schema={
        "block_number": pl.UInt64, "tx_hash": pl.Utf8, "log_index": pl.UInt32,
        "order_hash": pl.Utf8, "maker": pl.Utf8, "taker": pl.Utf8,
        "maker_asset_id": pl.Utf8, "taker_asset_id": pl.Utf8,
        "maker_amount_filled": pl.Utf8, "taker_amount_filled": pl.Utf8,
        "fee": pl.Utf8,
    })
    out = tmp_path / "order_filled_100_199.parquet"
    df.write_parquet(out)
    return tmp_path


def _fill(block, maker_aid, taker_aid, m_amt, t_amt, tx_suffix="1"):
    return {
        "block_number": block, "tx_hash": "0x" + tx_suffix * 64, "log_index": 0,
        "order_hash": "0x" + "a" * 64, "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "maker_asset_id": str(maker_aid), "taker_asset_id": str(taker_aid),
        "maker_amount_filled": str(m_amt), "taker_amount_filled": str(t_amt),
        "fee": "0",
    }


def test_load_onchain_trades_filters_by_token_ids(tmp_path, monkeypatch):
    from polydata.onchain.trades import load_onchain_trades

    fills = [
        # YES-token trade: taker buys with USDC → sign +1
        _fill(100, maker_aid=111, taker_aid=0,
              m_amt=1000 * 10**6, t_amt=600 * 10**6, tx_suffix="1"),
        # NO-token trade (different token) → filtered OUT when yes_id=111 only
        _fill(110, maker_aid=0, taker_aid=999,
              m_amt=400 * 10**6, t_amt=1000 * 10**6, tx_suffix="2"),
    ]
    slice_dir = _write_fake_slice(tmp_path, fills)

    def fake_rpc(*args, **kwargs):
        class _Stub:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def block_timestamp(self, bn): return 1_772_236_800 + (bn - 100) * 2
        return _Stub()
    monkeypatch.setattr("polydata.onchain.trades.PolygonRpcClient", fake_rpc)

    df = load_onchain_trades(
        market_id="mX",
        yes_token_id="111",
        no_token_id=None,  # only YES
        t_start=None, t_end=None,
        onchain_dir=slice_dir,
        rpc_url="https://mock",
    )
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["token_id"] == "111"
    assert r["sign"] == 1
    assert abs(r["price"] - 0.6) < 1e-9
    assert abs(r["size"] - 1000.0) < 1e-9
    assert r["market_id"] == "mX"


def test_sign_invariant_buyer_when_taker_posts_usdc(tmp_path, monkeypatch):
    from polydata.onchain.trades import load_onchain_trades

    # taker posted USDC (takerAssetId=0) → sign = +1
    fill = _fill(100, maker_aid=555, taker_aid=0,
                 m_amt=2000 * 10**6, t_amt=1000 * 10**6)
    _write_fake_slice(tmp_path, [fill])

    def fake_rpc(*args, **kwargs):
        class _S:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def block_timestamp(self, bn): return 1_772_236_800 + (bn - 100) * 2
        return _S()
    monkeypatch.setattr("polydata.onchain.trades.PolygonRpcClient", fake_rpc)

    df = load_onchain_trades(
        "m", yes_token_id="555", no_token_id=None,
        onchain_dir=tmp_path, rpc_url="mock",
    )
    assert df.row(0, named=True)["sign"] == 1


def test_sign_invariant_seller_when_maker_posts_usdc(tmp_path, monkeypatch):
    from polydata.onchain.trades import load_onchain_trades

    # maker posted USDC (makerAssetId=0) → sign = -1
    fill = _fill(100, maker_aid=0, taker_aid=777,
                 m_amt=1000 * 10**6, t_amt=2000 * 10**6)
    _write_fake_slice(tmp_path, [fill])

    def fake_rpc(*args, **kwargs):
        class _S:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def block_timestamp(self, bn): return 1_772_236_800 + (bn - 100) * 2
        return _S()
    monkeypatch.setattr("polydata.onchain.trades.PolygonRpcClient", fake_rpc)

    df = load_onchain_trades(
        "m", yes_token_id="777", no_token_id=None,
        onchain_dir=tmp_path, rpc_url="mock",
    )
    assert df.row(0, named=True)["sign"] == -1


def test_onchain_trade_stream_yields_in_time_order(tmp_path, monkeypatch):
    from polydata.onchain.trades import OnchainTradeStream

    fills = [
        _fill(101, maker_aid=111, taker_aid=0,
              m_amt=500 * 10**6, t_amt=300 * 10**6, tx_suffix="2"),
        _fill(100, maker_aid=111, taker_aid=0,
              m_amt=1000 * 10**6, t_amt=600 * 10**6, tx_suffix="1"),
    ]
    _write_fake_slice(tmp_path, fills)

    def fake_rpc(*args, **kwargs):
        class _S:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def block_timestamp(self, bn): return 1_772_236_800 + (bn - 100) * 2
        return _S()
    monkeypatch.setattr("polydata.onchain.trades.PolygonRpcClient", fake_rpc)

    stream = OnchainTradeStream(
        market_id="m", yes_token_id="111", no_token_id=None,
        onchain_dir=tmp_path, rpc_url="mock",
    )
    trades = list(stream)
    assert len(trades) == 2
    # block 100 before block 101
    assert trades[0].block_number == 100
    assert trades[1].block_number == 101


def test_load_onchain_trades_empty_onchain_dir_returns_empty_df(tmp_path):
    from polydata.onchain.trades import (
        ONCHAIN_TRADE_SCHEMA,
        load_onchain_trades,
    )
    # tmp_path is empty → no parquets
    df = load_onchain_trades(
        "m", yes_token_id="x", no_token_id=None,
        onchain_dir=tmp_path, rpc_url="mock",
    )
    assert df.height == 0
    assert set(df.columns) == set(ONCHAIN_TRADE_SCHEMA.keys())


def test_load_onchain_trades_both_token_ids_none_returns_empty(tmp_path):
    from polydata.onchain.trades import load_onchain_trades
    # Even with parquets present, both None → empty
    fill = _fill(100, maker_aid=111, taker_aid=0,
                 m_amt=1000 * 10**6, t_amt=600 * 10**6)
    _write_fake_slice(tmp_path, [fill])
    df = load_onchain_trades(
        "m", yes_token_id=None, no_token_id=None,
        onchain_dir=tmp_path, rpc_url="mock",
    )
    assert df.height == 0


def test_load_onchain_trades_unknown_token_returns_empty(tmp_path, monkeypatch):
    from polydata.onchain.trades import load_onchain_trades
    fill = _fill(100, maker_aid=111, taker_aid=0,
                 m_amt=1000 * 10**6, t_amt=600 * 10**6)
    _write_fake_slice(tmp_path, [fill])

    def fake_rpc(*args, **kwargs):
        class _S:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def block_timestamp(self, bn): return 1_772_236_800
        return _S()
    monkeypatch.setattr("polydata.onchain.trades.PolygonRpcClient", fake_rpc)

    df = load_onchain_trades(
        "m", yes_token_id="999_UNKNOWN", no_token_id=None,
        onchain_dir=tmp_path, rpc_url="mock",
    )
    assert df.height == 0


def test_attach_pre_trade_mid_uses_latest_sample_at_or_before_trade():
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    import polars as pl

    from polydata.onchain.trades import (
        ONCHAIN_TRADE_SCHEMA,
        attach_pre_trade_mid,
    )
    from polydata.resample import LOBSample

    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    samples = [
        LOBSample(t=t0, best_bid=Decimal("0.49"), best_ask=Decimal("0.51")),
        LOBSample(t=t0 + timedelta(seconds=5),
                  best_bid=Decimal("0.50"), best_ask=Decimal("0.52")),
    ]
    trades = pl.DataFrame(
        [{
            "market_id": "m", "t": t0 + timedelta(seconds=2),
            "block_number": 100, "tx_hash": "0x" + "1" * 64,
            "token_id": "111", "price": 0.51, "size": 100.0, "sign": 1,
            "maker": "0x" + "a" * 40, "taker": "0x" + "b" * 40,
        }],
        schema=ONCHAIN_TRADE_SCHEMA,
    )
    out = attach_pre_trade_mid(trades, samples)
    assert "mid_at_trade" in out.columns
    # t+2s → latest sample at/before is t0 → mid = (0.49+0.51)/2 = 0.50
    assert abs(out["mid_at_trade"][0] - 0.50) < 1e-9
