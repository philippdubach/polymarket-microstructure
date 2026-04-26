from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from polydata.events import PriceChange
from polydata.measures.latency import latency_distribution
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _record(ts_recv, ts_created):
    ev = PriceChange(
        update_type="price_change", market_id="m", token_id="t", side="YES",
        best_bid=Decimal("0.5"), best_ask=Decimal("0.51"),
        timestamp=ts_created.timestamp(),
        change_price=Decimal("0.5"), change_size=Decimal("10"),
        change_side="BUY",
    )
    return StreamRecord(ts_recv, ts_created, ev)


def test_latency_distribution_percentiles():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    # latency = ts_created - ts_received; fixture makes ts_received earlier
    # than ts_created by the target latency.
    records = [
        _record(t0 - timedelta(milliseconds=50), t0),     # 50 ms
        _record(t0 - timedelta(milliseconds=100), t0),    # 100 ms
        _record(t0 - timedelta(milliseconds=200), t0),    # 200 ms
        _record(t0 - timedelta(milliseconds=5000), t0),   # 5 s
    ]
    w = MeasurementWindow(
        market_id="m", t_start=t0 - timedelta(seconds=10), t_end=t0 + timedelta(seconds=1),
        events=records, sample_step=timedelta(seconds=1),
    )
    df = latency_distribution(w)
    assert df.schema == {
        "market_id": pl.Utf8, "n": pl.UInt32,
        "p50_ms": pl.Float64, "p90_ms": pl.Float64, "p99_ms": pl.Float64,
        "mean_ms": pl.Float64, "max_ms": pl.Float64,
    }
    row = df.row(0, named=True)
    assert row["n"] == 4
    assert 50 <= row["p50_ms"] <= 200
    assert row["max_ms"] == 5000
