from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import PriceChange
from polydata.measures.participants import mm_activity_signal
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _pc(t, size, side, change_side):
    # latency = ts_created - ts_received; make ts_received 20 ms earlier.
    return StreamRecord(
        t - timedelta(milliseconds=20), t,
        PriceChange(
            update_type="price_change", market_id="m", token_id="t", side=side,
            best_bid=Decimal("0.5"), best_ask=Decimal("0.51"),
            timestamp=t.timestamp(),
            change_price=Decimal("0.5"), change_size=Decimal(str(size)),
            change_side=change_side,
        ),
    )


def test_mm_activity_signal_fields():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = []
    for i in range(10):
        # alternating add (size 100) and cancel (size 0) with 20ms latency
        records.append(_pc(t0 + timedelta(milliseconds=100 * i),
                           100 if i % 2 == 0 else 0, "YES", "BUY"))
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=2),
        events=records, sample_step=timedelta(seconds=1),
    )
    df = mm_activity_signal(w)
    assert set(df.columns) == {
        "market_id", "n_events", "cancel_share", "fast_share_p50ms",
        "median_latency_ms", "quote_churn_per_sec",
    }
    row = df.row(0, named=True)
    assert row["n_events"] == 10
    assert abs(row["cancel_share"] - 0.5) < 0.01
    assert row["median_latency_ms"] <= 25
