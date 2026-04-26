from polydata.paths import parquet_files
from polydata.stream import MarketStream

REAL_MKT = "0x00000977017fa72fb6b1908ae694000d3b51f442c2552656b10bdbbfd16ff707"

def test_stream_yields_events_for_one_market():
    files = [parquet_files()[-50]]
    ms = MarketStream(market_id=REAL_MKT, files=files)
    events = list(ms)
    if events:
        assert all(e.event.market_id == REAL_MKT for e in events)
        for i in range(1, len(events)):
            assert events[i].ts_received >= events[i - 1].ts_received
    assert isinstance(events, list)

def test_stream_filters_by_side():
    files = [parquet_files()[-50]]
    yes = list(MarketStream(market_id=REAL_MKT, files=files, side="YES"))
    for r in yes:
        assert r.event.side == "YES"
