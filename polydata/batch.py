"""Batch runner for measure computation across markets.

compute_panel: for each market_id, build a MeasurementWindow, apply each
named measure, stack outputs with a `measure` column, and persist as a
single long-form parquet. Uses ProcessPoolExecutor for parallelism.

window_factory is injected for testability. The default factory (used in
scripts/compute_top_markets_panel.py) wraps polydata.stream.MarketStream
and the relevant slice of parquet_files().

Pickle / testability note
--------------------------
ProcessPoolExecutor requires all arguments to be pickleable. Closures
defined inside test functions are NOT pickleable. To keep tests fast and
hermetic, compute_panel uses a sequential fast-path when max_workers <= 1:
no executor is created and _run_one is called directly in a for-loop.
Pass max_workers=1 in unit tests. For production scripts pass max_workers=4
(or more) to exploit CPU-level parallelism across markets.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl

from polydata.window import MeasurementWindow

Measure = tuple[str, Callable[[MeasurementWindow], pl.DataFrame]]
WindowFactory = Callable[[str, datetime, datetime], MeasurementWindow]


def _run_one(
    market_id: str,
    t_start: datetime,
    t_end: datetime,
    measures: Sequence[Measure],
    window_factory: WindowFactory,
) -> pl.DataFrame:
    w = window_factory(market_id, t_start, t_end)
    frames: list[pl.DataFrame] = []
    for name, fn in measures:
        df = fn(w)
        if df.height == 0:
            continue
        df = df.with_columns(pl.lit(name).alias("measure"))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def compute_panel(
    market_ids: Sequence[str],
    t_start: datetime,
    t_end: datetime,
    measures: Sequence[Measure],
    window_factory: WindowFactory,
    out_path: Path,
    max_workers: int = 4,
) -> None:
    """Compute measures for each market and write a long-form parquet panel.

    Parameters
    ----------
    market_ids:
        Markets to process.
    t_start, t_end:
        Time range passed to window_factory for each market.
    measures:
        List of (name, callable) pairs. Each callable receives a
        MeasurementWindow and returns a pl.DataFrame.
    window_factory:
        Callable(market_id, t_start, t_end) -> MeasurementWindow.
        Must be pickleable when max_workers > 1.
    out_path:
        Destination parquet file. Parent dirs are created as needed.
    max_workers:
        Number of worker processes. Set to 1 for the sequential fast-path
        (required when window_factory is not pickleable, e.g. in tests).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames: list[pl.DataFrame] = []

    if max_workers <= 1:
        # Sequential fast-path: no pickling, safe for closures in tests.
        for mid in market_ids:
            frame = _run_one(mid, t_start, t_end, measures, window_factory)
            if frame.height > 0:
                all_frames.append(frame)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, mid, t_start, t_end, measures, window_factory): mid
                for mid in market_ids
            }
            for fut in as_completed(futures):
                frame = fut.result()
                if frame.height > 0:
                    all_frames.append(frame)

    panel = pl.concat(all_frames, how="diagonal_relaxed") if all_frames else pl.DataFrame()
    panel.write_parquet(out_path)
