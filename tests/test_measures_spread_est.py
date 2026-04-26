# tests/test_measures_spread_est.py
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polydata.events import BookSnapshot
from polydata.measures.spread_est import abdi_ranaldo_spread, roll_implied_spread
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


def _insufficient_window():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    records = [StreamRecord(t0, t0, _snap(t0, 0.49, 0.51))]
    return MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=3),
        events=records, sample_step=timedelta(seconds=1),
    )


def test_ar_returns_nan_below_minimum_trades():
    df = abdi_ranaldo_spread(_insufficient_window(), min_trades=100)
    row = df.row(0, named=True)
    assert row["n_trades"] < 100
    assert math.isnan(row["ar_half_spread_logodds"])
    assert row["insufficient_trades_flag"] is True


def test_roll_returns_nan_below_minimum_trades():
    df = roll_implied_spread(_insufficient_window(), min_trades=100)
    row = df.row(0, named=True)
    assert math.isnan(row["roll_half_spread_logodds"])
    assert row["insufficient_trades_flag"] is True


def test_ar_estimate_from_series_bounces_return_finite_spread():
    """Direct internal estimator test — bypasses trade inference.
    Bouncing mid series around 0.5 with known structure should give
    a finite non-negative half-spread and a CI that brackets the point.
    """
    import numpy as np

    from polydata.measures.spread_est import _ar_estimate_from_series
    rng = np.random.default_rng(0)
    mid = 0.5 + rng.choice([-0.005, 0.005], size=200) + rng.normal(0, 0.001, 200)
    mid = np.clip(mid, 0.01, 0.99)
    est = _ar_estimate_from_series(mid.tolist(), n_bootstrap=200, seed=0)
    assert est.point >= 0
    assert est.ci_lo <= est.point <= est.ci_hi
    assert est.n >= 100


def test_abdi_ranaldo_uses_injected_trades_when_provided():
    """Inject 150 synthetic trades alternating +/-0.01 around 0.5 to a window
    that would otherwise yield zero STRICT trades. AR should produce a finite
    half-spread on the injected stream."""
    import math
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    import polars as pl

    from polydata.events import BookSnapshot
    from polydata.measures.spread_est import abdi_ranaldo_spread
    from polydata.stream import StreamRecord
    from polydata.window import MeasurementWindow

    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    snap = BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal("0.499"), best_ask=Decimal("0.501"),
        timestamp=t0.timestamp(),
        bids=[(Decimal("0.499"), Decimal("10"))],
        asks=[(Decimal("0.501"), Decimal("10"))],
    )
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(minutes=5),
        events=[StreamRecord(t0, t0, snap)], sample_step=timedelta(seconds=1),
    )
    schema = {
        "market_id": pl.Utf8, "t": pl.Datetime(time_zone="UTC"),
        "token_id": pl.Utf8, "price": pl.Float64,
        "size": pl.Float64, "sign": pl.Int8,
    }
    rows = []
    for i in range(150):
        rows.append({
            "market_id": "m",
            "t": t0 + timedelta(seconds=i + 1),
            "token_id": "t",
            "price": 0.51 if i % 2 == 0 else 0.49,
            "size": 100.0,
            "sign": 1 if i % 2 == 0 else -1,
        })
    injected = pl.DataFrame(rows, schema=schema)
    df = abdi_ranaldo_spread(w, trades=injected, min_trades=100)
    row = df.row(0, named=True)
    assert row["n_trades"] == 150
    assert row["insufficient_trades_flag"] is False
    assert not math.isnan(row["ar_half_spread_logodds"])
