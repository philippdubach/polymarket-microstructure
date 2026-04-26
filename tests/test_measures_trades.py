# tests/test_measures_trades.py
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot, PriceChange
from polydata.measures.trades import (
    aggregate_trades_to_blocks,
    infer_trades_loose,
    infer_trades_strict,
)
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, bids, asks, token="t", side="YES"):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id=token, side=side,
        best_bid=Decimal(str(bids[0][0])) if bids else None,
        best_ask=Decimal(str(asks[0][0])) if asks else None,
        timestamp=t.timestamp(),
        bids=[(Decimal(str(p)), Decimal(str(q))) for p, q in bids],
        asks=[(Decimal(str(p)), Decimal(str(q))) for p, q in asks],
    )


def _pc(t, price, size, change_side, token="t", side="YES"):
    return PriceChange(
        update_type="price_change", market_id="m", token_id=token, side=side,
        best_bid=Decimal("0.5"), best_ask=Decimal("0.51"),
        timestamp=t.timestamp(),
        change_price=Decimal(str(price)), change_size=Decimal(str(size)),
        change_side=change_side,
    )


def _win(records, t_start, seconds):
    return MeasurementWindow(
        market_id="m", t_start=t_start, t_end=t_start + timedelta(seconds=seconds),
        events=records, sample_step=timedelta(seconds=1),
    )


def test_loose_signs_any_decrement():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, [(0.49, 10), (0.40, 5)], [(0.51, 10)])),
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _pc(t0 + timedelta(seconds=1), 0.40, 0, "BUY")),
    ]
    df = infer_trades_loose(_win(records, t0, 3))
    assert df.height == 1
    assert df["price"][0] == 0.40
    assert df["size"][0] == 5.0
    assert df["sign"][0] == -1


def test_strict_requires_top_of_book_and_co_timed_opposite_side():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, [(0.49, 10)], [(0.51, 10)])),
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _pc(t0 + timedelta(seconds=1), 0.51, 6, "SELL")),
        StreamRecord(t0 + timedelta(seconds=1, milliseconds=50),
                     t0 + timedelta(seconds=1, milliseconds=50),
                     _pc(t0 + timedelta(seconds=1, milliseconds=50), 0.49, 12, "BUY")),
    ]
    strict = infer_trades_strict(_win(records, t0, 3))
    assert strict.height == 1
    assert strict["price"][0] == 0.51
    assert strict["size"][0] == 4.0


def test_strict_rejects_mid_level_decrement():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, [(0.49, 10), (0.40, 5)], [(0.51, 10)])),
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _pc(t0 + timedelta(seconds=1), 0.40, 0, "BUY")),
    ]
    strict = infer_trades_strict(_win(records, t0, 3))
    assert strict.height == 0


def test_strict_top_n_levels_2_accepts_second_best():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    # Book: asks at [0.51: 10, 0.52: 5]. Decrement at 0.52 (2nd-best).
    # With top_n_levels=2 and co-timed bid companion, should produce 1 trade.
    records = [
        StreamRecord(t0, t0, _snap(t0, [(0.49, 10)], [(0.51, 10), (0.52, 5)])),
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _pc(t0 + timedelta(seconds=1), 0.52, 3, "SELL")),
        StreamRecord(t0 + timedelta(seconds=1, milliseconds=50),
                     t0 + timedelta(seconds=1, milliseconds=50),
                     _pc(t0 + timedelta(seconds=1, milliseconds=50), 0.49, 12, "BUY")),
    ]
    strict1 = infer_trades_strict(_win(records, t0, 3), top_n_levels=1)
    strict2 = infer_trades_strict(_win(records, t0, 3), top_n_levels=2)
    assert strict1.height == 0  # top-only rejects 2nd-best decrement
    assert strict2.height == 1  # top-2 accepts


def test_block_aggregation_groups_sub_block_events():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    trades = pl.DataFrame([
        {"market_id": "m", "t": t0, "token_id": "t",
         "price": 0.51, "size": 2.0, "sign": 1},
        {"market_id": "m", "t": t0 + timedelta(milliseconds=500),
         "token_id": "t", "price": 0.51, "size": 3.0, "sign": 1},
        {"market_id": "m", "t": t0 + timedelta(seconds=3),
         "token_id": "t", "price": 0.51, "size": 1.0, "sign": 1},
    ], schema={
        "market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
        "token_id": pl.Utf8, "price": pl.Float64, "size": pl.Float64, "sign": pl.Int8,
    })
    out = aggregate_trades_to_blocks(trades, block_seconds=2)
    assert out.height == 2
    assert out["size"].sum() == 6.0
