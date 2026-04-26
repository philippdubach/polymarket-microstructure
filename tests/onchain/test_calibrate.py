from polydata.onchain.calibrate import (
    CalibrationResult,
    run_precision_constrained_sweep,
)
from polydata.onchain.join import JoinResult


def _make_runner(table):
    """table maps (co, bs, ntop) -> (tp, fp, fn)."""
    def runner(co: int, bs: int, ntop: int) -> JoinResult:
        tp, fp, fn = table[(co, bs, ntop)]
        return JoinResult(
            n_inferred=tp + fp, n_onchain=tp + fn,
            n_true_positive=tp, n_false_positive=fp, n_false_negative=fn,
            size_mae=0.0, sign_agreement=1.0,
        )
    return runner


def test_selects_highest_recall_subject_to_precision_floor():
    # Full 2x2x1 = 4 cartesian cells. Precision floor 0.95.
    # A (50,1,1): P=0.95, R=0.60 -- ELIGIBLE
    # B (50,2,1): P~0.899 -- INELIGIBLE (P below floor)
    # C (100,1,1): P=0.95, R~0.704 -- ELIGIBLE, WINS (highest R)
    # D (100,2,1): P=1.00, R=0.50 -- ELIGIBLE, loses on recall
    table = {
        (50, 1, 1): (95, 5, 63),     # P=0.95, R=0.60
        (50, 2, 1): (80, 9, 20),     # P~0.899, below floor
        (100, 1, 1): (95, 5, 40),    # P=0.95, R~0.704
        (100, 2, 1): (50, 0, 50),    # P=1.00, R=0.50
    }
    result = run_precision_constrained_sweep(
        runner=_make_runner(table),
        co_timing_grid=[50, 100],
        block_seconds_grid=[1, 2],
        top_n_levels_grid=[1],
        precision_floor=0.95,
    )
    assert isinstance(result, CalibrationResult)
    assert (result.best_co_timing_ms, result.best_block_seconds,
            result.best_top_n_levels) == (100, 1, 1)
    assert result.grid.shape[0] == 4


def test_full_cartesian_produces_60_cells_for_production_grid_size():
    # Sanity: 5 x 4 x 3 = 60 cells for the production grid.
    table = {(co, bs, ntop): (10, 0, 0)
             for co in [50, 100, 200, 500, 1000]
             for bs in [1, 2, 3, 5]
             for ntop in [1, 2, 3]}
    result = run_precision_constrained_sweep(
        runner=_make_runner(table),
        co_timing_grid=[50, 100, 200, 500, 1000],
        block_seconds_grid=[1, 2, 3, 5],
        top_n_levels_grid=[1, 2, 3],
        precision_floor=0.95,
    )
    assert result.grid.shape[0] == 60


def test_falls_back_to_best_f1_when_no_cell_meets_floor():
    table = {
        (50, 1, 1): (80, 20, 10),   # P=0.80
        (100, 1, 1): (85, 15, 8),   # P~0.85
    }
    result = run_precision_constrained_sweep(
        runner=_make_runner(table),
        co_timing_grid=[50, 100], block_seconds_grid=[1],
        top_n_levels_grid=[1], precision_floor=0.95,
    )
    assert result.fallback_used is True
