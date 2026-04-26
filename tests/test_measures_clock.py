from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import PriceChange
from polydata.measures.clock import block_alignment
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _record(t):
    ev = PriceChange(
        update_type="price_change", market_id="m", token_id="t", side="YES",
        best_bid=Decimal("0.5"), best_ask=Decimal("0.51"),
        timestamp=t.timestamp(),
        change_price=Decimal("0.5"), change_size=Decimal("10"), change_side="BUY",
    )
    return StreamRecord(t, t, ev)


def test_block_alignment_shares():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    # Polygon block cadence ~2 s. Place 3 events within 100 ms of multiples
    # of 2 seconds relative to t0 (aligned), and 3 off-beat.
    aligned = [
        _record(t0 + timedelta(seconds=2, milliseconds=50)),
        _record(t0 + timedelta(seconds=4, milliseconds=20)),
        _record(t0 + timedelta(seconds=6, milliseconds=80)),
    ]
    off = [
        _record(t0 + timedelta(seconds=3)),
        _record(t0 + timedelta(seconds=5)),
        _record(t0 + timedelta(seconds=7)),
    ]
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=10),
        events=aligned + off, sample_step=timedelta(seconds=1),
    )
    r = block_alignment(w, block_period_ms=2000, tolerance_ms=100)
    row = r.row(0, named=True)
    assert row["n"] == 6
    assert 0.49 <= row["aligned_share"] <= 0.51
