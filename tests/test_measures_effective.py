# tests/test_measures_effective.py
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot, PriceChange
from polydata.measures.effective import effective_spread, realized_spread
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, bb, ba, bid_q=10, ask_q=10):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(bb)), best_ask=Decimal(str(ba)),
        timestamp=t.timestamp(),
        bids=[(Decimal(str(bb)), Decimal(str(bid_q)))],
        asks=[(Decimal(str(ba)), Decimal(str(ask_q)))],
    )


def _buy_at_ask(t, ask, remaining):
    return PriceChange(
        update_type="price_change", market_id="m", token_id="t", side="YES",
        best_bid=Decimal("0.49"), best_ask=Decimal(str(ask)),
        timestamp=t.timestamp(),
        change_price=Decimal(str(ask)), change_size=Decimal(str(remaining)),
        change_side="SELL",
    )


def _companion_bid(t, bid, remaining):
    return PriceChange(
        update_type="price_change", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(bid)), best_ask=Decimal("0.51"),
        timestamp=t.timestamp(),
        change_price=Decimal(str(bid)), change_size=Decimal(str(remaining)),
        change_side="BUY",
    )


def _win(records, t_start, seconds):
    return MeasurementWindow(
        market_id="m", t_start=t_start, t_end=t_start + timedelta(seconds=seconds),
        events=records, sample_step=timedelta(seconds=1),
    )


def _trade_scenario(t0, trade_price, trade_remaining):
    return [
        StreamRecord(t0, t0, _snap(t0, 0.49, trade_price)),
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _buy_at_ask(t0 + timedelta(seconds=1), trade_price, trade_remaining)),
        StreamRecord(t0 + timedelta(seconds=1, milliseconds=20),
                     t0 + timedelta(seconds=1, milliseconds=20),
                     _companion_bid(t0 + timedelta(seconds=1, milliseconds=20), 0.49, 12)),
    ]


def test_effective_spread_uses_pre_trade_mid():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = _trade_scenario(t0, 0.51, 6)
    df = effective_spread(_win(records, t0, 3), min_trades=1)
    required = {
        "market_id", "n_trades", "dollar_volume",
        "half_spread_prob_ew", "half_spread_prob_dw",
        "eff_spread_bps_mid_dw", "eff_spread_bps_minp_dw",
        "insufficient_trades_flag",
    }
    assert required.issubset(set(df.columns))
    row = df.row(0, named=True)
    assert row["n_trades"] == 1
    assert abs(row["half_spread_prob_dw"] - 0.01) < 1e-9
    # bps via min(p, 1-p): min(0.51, 0.49) = 0.49 → 2*0.01/0.49 * 10000 ≈ 408 bps
    assert 400 <= row["eff_spread_bps_minp_dw"] <= 420


def test_effective_spread_respects_min_trades():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = _trade_scenario(t0, 0.51, 6)
    df = effective_spread(_win(records, t0, 3), min_trades=30)
    row = df.row(0, named=True)
    assert row["n_trades"] == 1
    assert row["insufficient_trades_flag"] is True
    import math
    assert math.isnan(row["half_spread_prob_dw"])


def test_realized_spread_lag_sweep():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        *_trade_scenario(t0, 0.51, 6),
        StreamRecord(t0 + timedelta(seconds=5), t0 + timedelta(seconds=5),
                     _snap(t0 + timedelta(seconds=5), 0.49, 0.50)),
        StreamRecord(t0 + timedelta(seconds=10), t0 + timedelta(seconds=10),
                     _snap(t0 + timedelta(seconds=10), 0.48, 0.50)),
    ]
    df = realized_spread(_win(records, t0, 15), lags_sec=[1, 5, 10], min_trades=1)
    assert set(df["lag_sec"].to_list()) == {1, 5, 10}
    assert df.height == 3
    row5 = df.filter(pl.col("lag_sec") == 5).row(0, named=True)
    assert abs(row5["realized_half_spread_dw"] - 0.015) < 1e-9
    assert "price_impact_prob_dw" in df.columns


def test_effective_spread_uses_injected_trades_when_provided():
    """When `trades` is passed explicitly, internal STRICT inference is bypassed.
    We construct an empty window (STRICT would yield zero trades) plus an
    injected single-trade DataFrame and expect a non-zero effective-spread row."""
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    snap = BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal("0.49"), best_ask=Decimal("0.51"),
        timestamp=t0.timestamp(),
        bids=[(Decimal("0.49"), Decimal("10"))],
        asks=[(Decimal("0.51"), Decimal("10"))],
    )
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=5),
        events=[StreamRecord(t0, t0, snap)], sample_step=timedelta(seconds=1),
    )
    schema = {
        "market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    }
    injected = pl.DataFrame([{
        "market_id": "m", "t": t0 + timedelta(seconds=1),
        "token_id": "t", "price": 0.51, "size": 1000.0, "sign": 1,
    }], schema=schema)
    df = effective_spread(w, trades=injected, min_trades=1)
    row = df.row(0, named=True)
    assert row["n_trades"] == 1
    # Pre-trade mid = (0.49+0.51)/2 = 0.50; trade at 0.51 sign=+1 → half = 0.01
    assert abs(row["half_spread_prob_dw"] - 0.01) < 1e-9
