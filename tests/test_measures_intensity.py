from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import PriceChange
from polydata.measures.intensity import cancel_to_fill_ratio, quote_update_intensity
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _pc(t, p, q, side, best_bid=0.5, best_ask=0.51):
    return PriceChange(
        update_type="price_change", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(best_bid)), best_ask=Decimal(str(best_ask)),
        timestamp=t.timestamp(),
        change_price=Decimal(str(p)), change_size=Decimal(str(q)),
        change_side=side,
    )


def _win(records, t_start, seconds):
    return MeasurementWindow(
        market_id="m", t_start=t_start, t_end=t_start + timedelta(seconds=seconds),
        events=records, sample_step=timedelta(seconds=1),
    )


def test_quote_update_intensity_counts_events_per_bucket():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0 + timedelta(milliseconds=100 * i),
                     t0 + timedelta(milliseconds=100 * i),
                     _pc(t0 + timedelta(milliseconds=100 * i), 0.4, 10, "BUY"))
        for i in range(5)
    ]
    w = _win(records, t0, 10)
    df = quote_update_intensity(w, bucket=timedelta(seconds=1))
    assert df["n_updates"].sum() == 5
    assert df.filter(df["bucket_t"] == t0)["n_updates"].item() == 5


def test_cancel_to_fill_ratio_basics():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _pc(t0, 0.4, 10, "BUY")),    # add
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _pc(t0 + timedelta(seconds=1), 0.4, 0, "BUY")),  # cancel
    ]
    w = _win(records, t0, 2)
    r = cancel_to_fill_ratio(w)
    assert r.n_cancels == 1
    assert r.n_fills == 0
