from datetime import timedelta
from decimal import Decimal

from polydata.paths import parquet_files
from polydata.resample import resample_lob
from polydata.stream import MarketStream

REAL_MKT = "0x00000977017fa72fb6b1908ae694000d3b51f442c2552656b10bdbbfd16ff707"


def test_end_to_end_spread_series_for_one_market():
    files = parquet_files()[-24:]  # last ~24 hours
    records = list(MarketStream(market_id=REAL_MKT, files=files, side="YES"))
    if not records:
        import pytest
        pytest.skip("market not present in tail window")
    start = records[0].ts_received
    end = start + timedelta(hours=1)
    samples = resample_lob(records, start=start, end=end, step=timedelta(seconds=10))
    assert len(samples) == 360
    non_null = [s for s in samples if s.best_bid is not None and s.best_ask is not None]
    assert len(non_null) > 100, "expected non-null spread samples"
    spreads = [s.best_ask - s.best_bid for s in non_null]
    assert all(sp >= 0 for sp in spreads), "negative spread detected — LOB replay bug"
    assert max(spreads) <= Decimal("1"), "spread > 1 implies corrupt LOB"
