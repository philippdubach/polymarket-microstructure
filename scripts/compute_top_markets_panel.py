"""Pick the 3 most-active markets in the last 12 hours of the archive and
compute the full quote-based measure panel over the preceding 1 hour.
Emit a single parquet to artifacts/measures_panel_top3.parquet."""
from __future__ import annotations

from datetime import UTC, timedelta
from pathlib import Path

import polars as pl

from polydata.batch import Measure, compute_panel
from polydata.measures.clock import block_alignment
from polydata.measures.depth import mean_depth_by_level
from polydata.measures.effective import effective_spread, realized_spread
from polydata.measures.impact import amihud_illiquidity, kyle_lambda
from polydata.measures.intensity import quote_update_intensity
from polydata.measures.latency import latency_distribution
from polydata.measures.participants import mm_activity_signal
from polydata.measures.spread import price_conditional_spread, quoted_spread_series
from polydata.measures.spread_est import abdi_ranaldo_spread, roll_implied_spread
from polydata.measures.trades import infer_trades_strict
from polydata.paths import parquet_files
from polydata.stream import MarketStream
from polydata.window import MeasurementWindow


def _files_in_window(t_start, t_end):
    """Only keep parquet files whose filename-hour overlaps [t_start, t_end].

    Avoids scanning all 623 GB when the window is a single hour.
    Includes one file before t_start to preserve LOB state carried over
    from the prior hour (book_snapshots anchor the replay).
    """
    import re
    from datetime import datetime
    pat = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")
    result = []
    for f in parquet_files():
        m = pat.match(f.name)
        if not m:
            continue
        d, h = m.group(1), int(m.group(2))
        file_hour = datetime.fromisoformat(d).replace(hour=h, tzinfo=UTC)
        # Each file covers [file_hour, file_hour + 1h). Include if the hour
        # overlaps the window, or is the hour immediately before t_start
        # (to seed LOB state).
        file_end = file_hour + timedelta(hours=1)
        if file_end > t_start - timedelta(hours=1) and file_hour < t_end:
            result.append(f)
    return result


def default_window_factory(market_id, t_start, t_end):
    files = _files_in_window(t_start, t_end)
    records = list(MarketStream(market_id=market_id, files=files, side="YES"))
    records = [r for r in records if t_start <= r.ts_received < t_end]
    return MeasurementWindow(
        market_id=market_id, t_start=t_start, t_end=t_end,
        events=records, sample_step=timedelta(seconds=1),
    )


def _file_hour(path):
    """Extract the UTC hour-anchor from a parquet filename."""
    import re
    from datetime import datetime
    m = re.match(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet", path.name)
    assert m is not None, f"unrecognised filename {path.name}"
    d, h = m.group(1), int(m.group(2))
    return datetime.fromisoformat(d).replace(hour=h, tzinfo=UTC)


def pick_target_hour_file():
    """Pick the parquet file corresponding to the densest recent hour.

    Previous heuristic (top markets across last-12h) decoupled market
    selection from window selection; chosen markets had zero YES events
    in the target hour. We instead pick the single file with the highest
    row count in the last 48 hours of the archive — that's our in-window
    target — and draw top markets from it directly.
    """
    import pyarrow.parquet as pq
    candidates = parquet_files()[-48:]
    best_file = None
    best_rows = -1
    for f in candidates:
        rows = pq.read_metadata(f).num_rows
        if rows > best_rows:
            best_rows = rows
            best_file = f
    assert best_file is not None
    return best_file, best_rows


def pick_top_markets_in_file(target_file, n: int = 3) -> list[str]:
    """Top-N markets by row count IN the target file (YES+NO aggregated)."""
    return (
        pl.scan_parquet(target_file)
        .group_by("market_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .limit(n)
        .collect(engine="streaming")["market_id"]
        .to_list()
    )


if __name__ == "__main__":
    target_file, target_rows = pick_target_hour_file()
    t_start = _file_hour(target_file)
    t_end = t_start + timedelta(hours=1)
    ids = pick_top_markets_in_file(target_file, n=3)
    print(f"target file: {target_file.name} ({target_rows:,} rows)")
    print(f"window: {t_start} -> {t_end}")
    print(f"markets: {ids}")
    measures: list[Measure] = [
        # Quote-based (Plan 2a)
        ("quoted_spread_series", quoted_spread_series),
        ("price_conditional_spread", lambda w: price_conditional_spread(w, n_bins=20)),
        ("depth_top_k", mean_depth_by_level),
        ("quote_update_intensity", quote_update_intensity),
        ("latency_distribution", latency_distribution),
        ("block_alignment", block_alignment),
        ("mm_activity_signal", mm_activity_signal),
        # Trade-based (Plan 2b, expert-refined)
        ("trades_strict", infer_trades_strict),
        ("effective_spread", effective_spread),
        ("realized_spread", realized_spread),
        ("abdi_ranaldo_spread", abdi_ranaldo_spread),
        ("roll_implied_spread_diagnostic", roll_implied_spread),
        ("kyle_lambda_1min", kyle_lambda),
        ("amihud_illiquidity_1min", amihud_illiquidity),
    ]
    out = Path("artifacts/measures_panel_top3.parquet")
    compute_panel(
        market_ids=ids,
        t_start=t_start, t_end=t_end,
        measures=measures,
        window_factory=default_window_factory,
        out_path=out,
        max_workers=1,  # sequential — parquet I/O dominates
    )
    panel = pl.read_parquet(out)
    print(f"panel shape: {panel.shape}")
    print(panel.group_by("measure").agg(pl.len().alias("rows")).sort("rows", descending=True))
