"""Plan 3d-bis T2: sample-step sensitivity on top-5 markets.

For each (market, sample_step, measure) cell we record the headline
output column. A reviewer can read off whether the 60s default we
used in T3 is a material accuracy compromise on top markets.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from polydata.events import parse_event
from polydata.measures.effective import effective_spread
from polydata.measures.impact import amihud_illiquidity, kyle_lambda
from polydata.measures.spread_est import abdi_ranaldo_spread, roll_implied_spread
from polydata.onchain.token_map import load_token_map
from polydata.stream import StreamRecord
from polydata.window import MeasurementWindow

SAMPLE_STEPS_S = [1, 10, 60, 300]

MEASURES = [
    ("effective_spread", effective_spread, "half_spread_prob_dw"),
    ("abdi_ranaldo_spread", abdi_ranaldo_spread, "ar_half_spread_logodds"),
    ("roll_implied_spread", roll_implied_spread, "roll_half_spread_logodds"),
    ("kyle_lambda", kyle_lambda, "kyle_lambda_logodds"),
    ("amihud_illiquidity", amihud_illiquidity, "amihud_ratio_of_means"),
]


def _load_market_window(
    market_id: str, t_start: datetime, t_end: datetime,
) -> list[StreamRecord]:
    slice_df = (
        pl.scan_parquet("data/panel_orderbook_shards/*.parquet")
        .filter(pl.col("market_id") == market_id)
        .filter(
            (pl.col("timestamp_received") >= t_start)
            & (pl.col("timestamp_received") < t_end)
        )
        .sort("timestamp_received")
        .collect()
    )
    out: list[StreamRecord] = []
    for row in slice_df.iter_rows(named=True):
        try:
            ev = parse_event(row["data"])
        except Exception:
            continue
        if ev.side != "YES":
            continue
        out.append(StreamRecord(
            ts_received=row["timestamp_received"],
            ts_created=row["timestamp_created_at"],
            event=ev,
        ))
    return out


def main() -> int:
    panel = pl.read_parquet("data/panel.parquet").filter(
        pl.col("stratum") == "top"
    ).head(5)
    tm = load_token_map()
    panel = panel.join(tm, on="market_id", how="inner")
    print(f"sensitivity cohort: {panel.height} markets", flush=True)
    t_start = datetime(2026, 3, 13, tzinfo=UTC)
    t_end = datetime(2026, 3, 14, tzinfo=UTC)

    agg = pl.read_parquet("data/panel_onchain_cache.parquet")
    agg_window = agg.filter(
        (pl.col("t") >= t_start) & (pl.col("t") < t_end)
    )

    rows: list[dict] = []
    for r in panel.iter_rows(named=True):
        recs = _load_market_window(r["market_id"], t_start, t_end)
        onchain = agg_window.filter(
            pl.col("token_id") == r["yes_token_id"]
        ).with_columns(pl.lit(r["market_id"]).alias("market_id")).select([
            "market_id", "t", "token_id", "price", "size", "sign",
        ])
        print(
            f"\n{r['market_id'][:14]}… events={len(recs):,} "
            f"trades={onchain.height:,}", flush=True,
        )
        for step in SAMPLE_STEPS_S:
            print(f"  step={step}s …", flush=True)
            w = MeasurementWindow(
                market_id=r["market_id"], t_start=t_start, t_end=t_end,
                events=recs, sample_step=timedelta(seconds=step),
            )
            for name, fn, value_col in MEASURES:
                df = fn(w, trades=onchain)
                if df.height == 0:
                    rows.append({
                        "market_id": r["market_id"],
                        "sample_step_s": step,
                        "measure": name,
                        "value": None,
                    })
                    continue
                vals = df[value_col].to_list()
                rows.append({
                    "market_id": r["market_id"],
                    "sample_step_s": step,
                    "measure": name,
                    "value": vals[0] if vals else None,
                })
    out = pl.DataFrame(rows)
    out_path = Path("artifacts/sample_step_sensitivity.parquet")
    out.write_parquet(out_path)
    print(f"\nwrote {out.height:,} rows to {out_path}", flush=True)
    pivot = out.pivot(
        values="value", index=["market_id", "measure"],
        on="sample_step_s",
    )
    pl.Config.set_fmt_str_lengths(80)
    print(pivot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
