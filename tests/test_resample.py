from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import BookSnapshot
from polydata.resample import resample_lob
from polydata.stream import StreamRecord


def _snap(t, bids, asks):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(bids[0][0])), best_ask=Decimal(str(asks[0][0])),
        timestamp=t.timestamp(),
        bids=[(Decimal(str(p)), Decimal(str(q))) for p, q in bids],
        asks=[(Decimal(str(p)), Decimal(str(q))) for p, q in asks],
    )


def test_resample_forward_fills_between_events():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, [(0.5, 10)], [(0.51, 10)])),
        StreamRecord(t0 + timedelta(seconds=5), t0 + timedelta(seconds=5),
                     _snap(t0 + timedelta(seconds=5), [(0.52, 10)], [(0.53, 10)])),
    ]
    out = resample_lob(records, start=t0, end=t0 + timedelta(seconds=10),
                       step=timedelta(seconds=1))
    assert len(out) == 10
    assert out[0].best_bid == Decimal("0.5")
    assert out[4].best_bid == Decimal("0.5")
    assert out[5].best_bid == Decimal("0.52")
    assert out[9].best_bid == Decimal("0.52")


def test_resample_empty_before_first_event():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [StreamRecord(t0 + timedelta(seconds=5), t0 + timedelta(seconds=5),
                            _snap(t0 + timedelta(seconds=5), [(0.5, 10)], [(0.51, 10)]))]
    out = resample_lob(records, start=t0, end=t0 + timedelta(seconds=10),
                       step=timedelta(seconds=1))
    assert out[0].best_bid is None
    assert out[5].best_bid == Decimal("0.5")
