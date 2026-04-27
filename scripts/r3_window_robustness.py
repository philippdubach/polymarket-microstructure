"""R3 — Round 1 peer-review required revision #3.

Compute STRICT-vs-on-chain effective-spread and Kyle's λ sign-flip
rates on window-A (2026-03-07 → 03-14) and window-B
(2026-03-14 → 03-21) and report the deltas. Used to demonstrate
that the §7.2 headline rates are not window-specific.

Reads:
  - artifacts/measures_compare_top100_window_a.parquet (saved 2026-04-26)
  - artifacts/measures_compare_top100_window_b.parquet (R3 rerun)

Writes:
  - artifacts/r3_window_robustness.csv
  - stdout summary suitable for the paper §7.2 robustness paragraph
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

WINDOW_A = Path("artifacts/measures_compare_top100_window_a.parquet")
WINDOW_B = Path("artifacts/measures_compare_top100_window_b.parquet")
OUT_CSV = Path("artifacts/r3_window_robustness.csv")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return centre - half, centre + half


def signflip_rate(
    df: pl.DataFrame, measure: str, value_col: str,
) -> tuple[int, int, float, float, float]:
    sub = df.filter(pl.col("measure") == measure)
    pivot = sub.pivot(values=value_col, index="market_id", on="trade_source")
    if "strict" not in pivot.columns or "onchain" not in pivot.columns:
        return 0, 0, float("nan"), float("nan"), float("nan")
    keep = pivot.filter(
        pl.col("strict").is_finite() & pl.col("onchain").is_finite()
    )
    n = keep.height
    if n == 0:
        return 0, 0, float("nan"), float("nan"), float("nan")
    flips = (keep["strict"].sign() != keep["onchain"].sign()).sum()
    rate = flips / n
    lo, hi = wilson(int(flips), n)
    return int(flips), n, rate, lo, hi


def main() -> int:
    if not WINDOW_B.exists():
        print(f"window-B parquet not yet present: {WINDOW_B}", flush=True)
        print("R3 compute still running. Re-run this script after it "
              "completes.", flush=True)
        return 1
    rows: list[dict] = []
    for label, path in (("A 2026-03-07_03-14", WINDOW_A),
                        ("B 2026-03-14_03-21", WINDOW_B)):
        df = pl.read_parquet(path)
        for measure, vcol in (("effective_spread", "half_spread_prob_dw"),
                              ("kyle_lambda", "kyle_lambda_logodds")):
            k, n, p, lo, hi = signflip_rate(df, measure, vcol)
            rows.append({
                "window": label, "measure": measure,
                "k_flip": k, "n": n,
                "rate": p, "ci_lo": lo, "ci_hi": hi,
            })
    out = pl.DataFrame(rows)
    out.write_csv(OUT_CSV)
    print(out)

    print("\n--- Side-by-side ---", flush=True)
    for measure in ("effective_spread", "kyle_lambda"):
        a = out.filter(
            (pl.col("measure") == measure)
            & (pl.col("window").str.starts_with("A"))
        ).row(0, named=True)
        b = out.filter(
            (pl.col("measure") == measure)
            & (pl.col("window").str.starts_with("B"))
        ).row(0, named=True)
        delta = b["rate"] - a["rate"]
        within_5pp = abs(delta) <= 0.05
        print(
            f"{measure:>20s}  A: {a['k_flip']}/{a['n']} = {a['rate']:.3f}  "
            f"B: {b['k_flip']}/{b['n']} = {b['rate']:.3f}  "
            f"Δ = {delta:+.3f}  within 5pp? {within_5pp}",
            flush=True,
        )
    print(f"\nwrote {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
