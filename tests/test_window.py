from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from polydata.events import BookSnapshot
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, bids, asks, side="YES"):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side=side,
        best_bid=Decimal(str(bids[0][0])) if bids else None,
        best_ask=Decimal(str(asks[0][0])) if asks else None,
        timestamp=t.timestamp(),
        bids=[(Decimal(str(p)), Decimal(str(q))) for p, q in bids],
        asks=[(Decimal(str(p)), Decimal(str(q))) for p, q in asks],
    )


def test_window_holds_events_and_samples():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    evt = _snap(t0, [(0.5, 10)], [(0.51, 10)])
    records = [StreamRecord(t0, t0, evt)]
    w = MeasurementWindow(
        market_id="m",
        t_start=t0,
        t_end=t0 + timedelta(seconds=10),
        events=records,
        sample_step=timedelta(seconds=1),
    )
    assert w.market_id == "m"
    assert len(w.events) == 1
    assert len(w.samples) == 10
    assert w.samples[0].best_bid == Decimal("0.5")


def test_window_rejects_empty_range():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        MeasurementWindow(
            market_id="m", t_start=t0, t_end=t0,
            events=[], sample_step=timedelta(seconds=1),
        )
