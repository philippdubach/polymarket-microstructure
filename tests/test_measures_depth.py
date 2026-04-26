from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot
from polydata.measures.depth import depth_series, mean_depth_by_level
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow


def _snap(t, bids, asks):
    return BookSnapshot(
        update_type="book_snapshot", market_id="m", token_id="t", side="YES",
        best_bid=Decimal(str(bids[0][0])), best_ask=Decimal(str(asks[0][0])),
        timestamp=t.timestamp(),
        bids=[(Decimal(str(p)), Decimal(str(q))) for p, q in bids],
        asks=[(Decimal(str(p)), Decimal(str(q))) for p, q in asks],
    )


def test_depth_series_top_k_columns():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    evt = _snap(t0, [(0.5, 10), (0.49, 5)], [(0.51, 8), (0.52, 3)])
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=2),
        events=[StreamRecord(t0, t0, evt)], sample_step=timedelta(seconds=1),
    )
    df = depth_series(w, ks=(1, 2, 5))
    expected_cols = {
        "market_id", "t",
        "bid_top1", "ask_top1",
        "bid_top2", "ask_top2",
        "bid_top5", "ask_top5",
    }
    assert set(df.columns) >= expected_cols
    assert df["bid_top1"][0] == 10.0
    assert df["bid_top2"][0] == 15.0
    assert df["ask_top2"][0] == 11.0
    assert df["bid_top5"][0] == 15.0


def test_mean_depth_by_level():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    evt = _snap(t0, [(0.5, 10), (0.49, 5)], [(0.51, 8)])
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=2),
        events=[StreamRecord(t0, t0, evt)], sample_step=timedelta(seconds=1),
    )
    df = mean_depth_by_level(w, ks=(1, 2, 5))
    assert df.schema == {"market_id": pl.Utf8, "k": pl.UInt32,
                         "mean_bid_depth": pl.Float64, "mean_ask_depth": pl.Float64}
    assert df.filter(pl.col("k") == 1)["mean_bid_depth"].item() == 10.0
    assert df.filter(pl.col("k") == 2)["mean_bid_depth"].item() == 15.0


def test_depth_series_supports_k_3():
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    evt = _snap(t0, [(0.5, 10), (0.49, 5), (0.48, 2), (0.47, 1)],
                    [(0.51, 8), (0.52, 3), (0.53, 1)])
    w = MeasurementWindow(
        market_id="m", t_start=t0, t_end=t0 + timedelta(seconds=1),
        events=[StreamRecord(t0, t0, evt)], sample_step=timedelta(seconds=1),
    )
    df = depth_series(w, ks=(1, 3, 5))
    assert df["bid_top3"][0] == 17.0
    assert df["ask_top3"][0] == 12.0
