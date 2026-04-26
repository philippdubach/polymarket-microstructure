import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import BookSnapshot
from polydata.measures.impact import amihud_illiquidity, kyle_lambda
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, bb, ba):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(bb)), best_ask=Decimal(str(ba)),
        timestamp=t.timestamp(),
        bids=[(Decimal(str(bb)), Decimal("100"))],
        asks=[(Decimal(str(ba)), Decimal("100"))],
    )


def test_kyle_schema_and_nan_on_empty():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [StreamRecord(t0, t0, _snap(t0, 0.49, 0.51))]
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(minutes=10),
        events=records, sample_step=timedelta(seconds=1),
    )
    df = kyle_lambda(w, bucket=timedelta(minutes=1))
    required = {
        "market_id", "n_buckets",
        "kyle_lambda_logodds", "kyle_lambda_price",
        "kyle_intercept", "kyle_r2",
        "kyle_se_hac", "kyle_se_hc3",
        "insufficient_buckets_flag",
    }
    assert required.issubset(set(df.columns))
    row = df.row(0, named=True)
    assert row["insufficient_buckets_flag"] is True
    assert math.isnan(row["kyle_lambda_logodds"])


def test_amihud_reports_both_estimators():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [StreamRecord(t0, t0, _snap(t0, 0.49, 0.51))]
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(minutes=10),
        events=records, sample_step=timedelta(seconds=1),
    )
    df = amihud_illiquidity(w, bucket=timedelta(minutes=1))
    required = {
        "market_id", "n_buckets",
        "amihud_ratio_of_means",
        "amihud_mean_of_ratios",
        "insufficient_buckets_flag",
    }
    assert required.issubset(set(df.columns))
    row = df.row(0, named=True)
    assert row["insufficient_buckets_flag"] is True


def test_kyle_lambda_accepts_injected_trades():
    """Inject 30 synthetic trades to a window with monotonic mid drift.
    Kyle's lambda should produce a finite slope on the injected stream."""
    import math
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    import polars as pl

    from polydata.events import BookSnapshot
    from polydata.measures.impact import kyle_lambda
    from polydata.stream import StreamRecord
    from polydata.window import MeasurementWindow

    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    snap_events = []
    for i in range(31):
        ts = t0 + timedelta(minutes=i)
        snap_events.append(StreamRecord(ts, ts, BookSnapshot(
            update_type="book_snapshot", market_id="m", token_id="t", side="YES",
            best_bid=Decimal(str(0.5 - 0.001 * i)),
            best_ask=Decimal(str(0.502 - 0.001 * i)),
            timestamp=ts.timestamp(),
            bids=[(Decimal(str(0.5 - 0.001 * i)), Decimal("10"))],
            asks=[(Decimal(str(0.502 - 0.001 * i)), Decimal("10"))],
        )))
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(minutes=31),
        events=snap_events, sample_step=timedelta(seconds=1),
    )
    schema = {
        "market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    }
    injected = pl.DataFrame(
        [{
            "market_id": "m",
            "t": t0 + timedelta(minutes=i, seconds=30),
            "token_id": "t",
            "price": 0.501 - 0.001 * i,
            "size": 100.0,
            "sign": 1 if i % 2 == 0 else -1,
        } for i in range(30)],
        schema=schema,
    )
    df = kyle_lambda(w, trades=injected, bucket=timedelta(minutes=1), min_buckets=10)
    row = df.row(0, named=True)
    assert not math.isnan(row["kyle_lambda_logodds"])


def test_amihud_accepts_injected_trades():
    import math
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    import polars as pl

    from polydata.events import BookSnapshot
    from polydata.measures.impact import amihud_illiquidity
    from polydata.stream import StreamRecord
    from polydata.window import MeasurementWindow

    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    snap_events = []
    for i in range(31):
        ts = t0 + timedelta(minutes=i)
        snap_events.append(StreamRecord(ts, ts, BookSnapshot(
            update_type="book_snapshot", market_id="m", token_id="t", side="YES",
            best_bid=Decimal(str(0.5 - 0.001 * i)),
            best_ask=Decimal(str(0.502 - 0.001 * i)),
            timestamp=ts.timestamp(),
            bids=[(Decimal(str(0.5 - 0.001 * i)), Decimal("10"))],
            asks=[(Decimal(str(0.502 - 0.001 * i)), Decimal("10"))],
        )))
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(minutes=31),
        events=snap_events, sample_step=timedelta(seconds=1),
    )
    schema = {
        "market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    }
    injected = pl.DataFrame(
        [{
            "market_id": "m",
            "t": t0 + timedelta(minutes=i, seconds=30),
            "token_id": "t",
            "price": 0.501 - 0.001 * i,
            "size": 100.0,
            "sign": 1 if i % 2 == 0 else -1,
        } for i in range(30)],
        schema=schema,
    )
    df = amihud_illiquidity(w, trades=injected, bucket=timedelta(minutes=1), min_buckets=10)
    row = df.row(0, named=True)
    assert not math.isnan(row["amihud_ratio_of_means"])
