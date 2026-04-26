from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot
from polydata.measures.spread import price_conditional_spread, quoted_spread_series
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, bb, ba):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(bb)), best_ask=Decimal(str(ba)),
        timestamp=t.timestamp(),
        bids=[(Decimal(str(bb)), Decimal("10"))],
        asks=[(Decimal(str(ba)), Decimal("10"))],
    )


def _win(records, t_start, seconds):
    return MeasurementWindow(
        market_id="m", t_start=t_start, t_end=t_start + timedelta(seconds=seconds),
        events=records, sample_step=timedelta(seconds=1),
    )


def test_quoted_spread_series_schema_and_values():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, 0.50, 0.52)),
        StreamRecord(t0 + timedelta(seconds=3), t0 + timedelta(seconds=3),
                     _snap(t0 + timedelta(seconds=3), 0.55, 0.56)),
    ]
    w = _win(records, t0, 6)
    df = quoted_spread_series(w)
    assert df.schema == {"market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
                         "mid": pl.Float64, "half_spread": pl.Float64}
    assert df.height == 6
    first_half = df["half_spread"][0]
    assert abs(first_half - 0.01) < 1e-9
    last_half = df["half_spread"][5]
    assert abs(last_half - 0.005) < 1e-9


def test_quoted_spread_ignores_one_sided_samples():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [StreamRecord(t0 + timedelta(seconds=2), t0 + timedelta(seconds=2),
                            _snap(t0 + timedelta(seconds=2), 0.5, 0.51))]
    w = _win(records, t0, 5)
    df = quoted_spread_series(w)
    non_null = df.filter(pl.col("half_spread").is_not_null())
    assert non_null.height == 3


def test_price_conditional_spread_bins():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [
        StreamRecord(t0, t0, _snap(t0, 0.05, 0.06)),  # mid 0.055 — longshot
        StreamRecord(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1),
                     _snap(t0 + timedelta(seconds=1), 0.49, 0.51)),  # mid 0.5
    ]
    w = _win(records, t0, 4)
    df = price_conditional_spread(w, n_bins=10)
    assert set(df.columns) == {"market_id", "price_bin_lo", "price_bin_hi", "n", "mean_half_spread"}
    assert df.height <= 10
    assert df["n"].sum() > 0
