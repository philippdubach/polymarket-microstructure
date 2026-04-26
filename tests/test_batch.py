import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polydata.batch import compute_panel


def test_compute_panel_writes_parquet():
    import polars as pl

    from polydata.measures.latency import latency_distribution

    def fake_window_factory(market_id, t_start, t_end):
        from decimal import Decimal

        from polydata.events import BookSnapshot
        from polydata.stream import StreamRecord
        from polydata.window import MeasurementWindow

        evt = BookSnapshot(
            update_type="book_snapshot", market_id=market_id, token_id="t", side="YES",
            best_bid=Decimal("0.5"), best_ask=Decimal("0.51"),
            timestamp=t_start.timestamp(),
            bids=[(Decimal("0.5"), Decimal("10"))],
            asks=[(Decimal("0.51"), Decimal("10"))],
        )
        return MeasurementWindow(
            market_id=market_id, t_start=t_start, t_end=t_end,
            events=[StreamRecord(t_start, t_start, evt)],
            sample_step=timedelta(seconds=1),
        )

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "panel.parquet"
        t0 = datetime(2026, 3, 1, tzinfo=UTC)
        compute_panel(
            market_ids=["m1", "m2"],
            t_start=t0, t_end=t0 + timedelta(seconds=5),
            measures=[("latency", latency_distribution)],
            window_factory=fake_window_factory,
            out_path=out,
            max_workers=1,
        )
        assert out.exists()
        df = pl.read_parquet(out)
        assert set(df["market_id"].unique().to_list()) == {"m1", "m2"}
        assert "measure" in df.columns
