from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot
from polydata.measures._trade_source import resolve_trades
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _empty_window():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    snap = BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal("0.5"), best_ask=Decimal("0.51"),
        timestamp=t0.timestamp(),
        bids=[(Decimal("0.5"), Decimal("10"))],
        asks=[(Decimal("0.51"), Decimal("10"))],
    )
    return MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=2),
        events=[StreamRecord(t0, t0, snap)], sample_step=timedelta(seconds=1),
    )


def test_resolve_trades_returns_injected_when_provided():
    schema = {
        "market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    }
    injected = pl.DataFrame([{
        "market_id": "m", "t": datetime(2026, 3, 1, tzinfo=UTC),
        "token_id": "t", "price": 0.5, "size": 100.0, "sign": 1,
    }], schema=schema)
    out = resolve_trades(_empty_window(), injected)
    assert out.height == 1
    assert out["price"][0] == 0.5
    assert out["sign"][0] == 1


def test_resolve_trades_falls_back_to_strict_when_none():
    out = resolve_trades(_empty_window(), None)
    assert isinstance(out, pl.DataFrame)
    expected_cols = {"market_id", "t", "token_id", "price", "size", "sign"}
    assert expected_cols.issubset(set(out.columns))


def test_strict_and_onchain_share_canonical_schema():
    """STRICT inferred trades and on-chain authoritative trades must share
    the schema columns measures consume."""
    from polydata.measures.trades import _TRADE_SCHEMA as STRICT_SCHEMA
    from polydata.onchain.trades import ONCHAIN_TRADE_SCHEMA

    required = {"market_id", "t", "token_id", "price", "size", "sign"}
    assert required.issubset(set(STRICT_SCHEMA.keys()))
    assert required.issubset(set(ONCHAIN_TRADE_SCHEMA.keys()))
    for col in required:
        assert STRICT_SCHEMA[col] == ONCHAIN_TRADE_SCHEMA[col], (
            f"type mismatch on {col}: "
            f"STRICT={STRICT_SCHEMA[col]} ONCHAIN={ONCHAIN_TRADE_SCHEMA[col]}"
        )
