from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import BookSnapshot
from polydata.measures.consistency import yes_no_parity_check
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, token, side, bb, ba):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id=token, side=side,
        best_bid=Decimal(str(bb)), best_ask=Decimal(str(ba)),
        timestamp=t.timestamp(),
        bids=[(Decimal(str(bb)), Decimal("10"))],
        asks=[(Decimal(str(ba)), Decimal("10"))],
    )


def test_parity_close_to_one():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, "yes", "YES", 0.49, 0.51)),
        StreamRecord(t0, t0, _snap(t0, "no", "NO", 0.49, 0.51)),
    ]
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=5),
        events=records, sample_step=timedelta(seconds=1),
    )
    df = yes_no_parity_check(w, yes_token="yes", no_token="no")
    row = df.row(0, named=True)
    assert 0.99 <= row["mean_parity_sum"] <= 1.01
    assert row["n_samples"] > 0
