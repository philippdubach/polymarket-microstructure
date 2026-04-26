"""3D threshold sweep + precision-constrained selection.

Selection rule (per docs/preregistration_2c.md): among cells with
precision >= precision_floor, pick the one with highest recall; break
ties by F1 then by fewer FPs. If no cell meets the floor, fall back
to best F1 and flag `fallback_used`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from polydata.onchain.join import JoinResult, compute_metrics

RunnerFn = Callable[[int, int, int], JoinResult]


@dataclass(frozen=True)
class CalibrationResult:
    best_co_timing_ms: int
    best_block_seconds: int
    best_top_n_levels: int
    best_precision: float
    best_recall: float
    best_f1: float
    fallback_used: bool
    grid: pl.DataFrame


def run_precision_constrained_sweep(
    runner: RunnerFn,
    co_timing_grid: list[int],
    block_seconds_grid: list[int],
    top_n_levels_grid: list[int],
    precision_floor: float = 0.95,
) -> CalibrationResult:
    rows: list[dict] = []
    for co in co_timing_grid:
        for bs in block_seconds_grid:
            for ntop in top_n_levels_grid:
                m = compute_metrics(runner(co, bs, ntop))
                rows.append({
                    "co_timing_ms": co, "block_seconds": bs, "top_n_levels": ntop,
                    "precision": m["precision"], "recall": m["recall"],
                    "f1": m["f1"], "size_mae": m["size_mae"],
                    "sign_agreement": m["sign_agreement"], "n_tp": m["n_tp"],
                    "n_fp": m["n_fp"], "n_fn": m["n_fn"],
                })
    grid = pl.DataFrame(rows)
    eligible = grid.filter(pl.col("precision") >= precision_floor)
    fallback_used = eligible.height == 0
    if fallback_used:
        winner = grid.sort(["f1", "n_fp"], descending=[True, False]).row(0, named=True)
    else:
        winner = eligible.sort(
            ["recall", "f1", "n_fp"], descending=[True, True, False],
        ).row(0, named=True)

    grid = grid.with_columns(
        ((pl.col("co_timing_ms") == winner["co_timing_ms"])
         & (pl.col("block_seconds") == winner["block_seconds"])
         & (pl.col("top_n_levels") == winner["top_n_levels"]))
        .alias("selected_flag")
    )

    return CalibrationResult(
        best_co_timing_ms=int(winner["co_timing_ms"]),
        best_block_seconds=int(winner["block_seconds"]),
        best_top_n_levels=int(winner["top_n_levels"]),
        best_precision=float(winner["precision"]),
        best_recall=float(winner["recall"]),
        best_f1=float(winner["f1"]),
        fallback_used=fallback_used,
        grid=grid,
    )
