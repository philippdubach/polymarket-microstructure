# Plan 3c pre-registration

**Date:** 2026-04-24
**Status:** LOCKED — any change requires explicit paper-spec amendment + git log entry

## Panel definition

- **Top-100 stratum:** the 100 distinct `market_id` values with the highest
  on-chain USDC trade volume across the scrape window
  2026-02-28 00:00 UTC → 2026-03-27 23:59 UTC.
  - Volume = `sum(size * price)` across YES and NO tokens of the market.
  - Ties broken by on-chain trade count (descending), then `market_id`
    lexicographic.
  - Restriction: market must resolve to a `(yes_token_id, no_token_id)`
    pair via `data/clob_token_map.parquet`. With the ~934k-row CLOB cache
    this is not a binding constraint.
- **Random-500 stratum:** 500 distinct `market_id` values sampled uniformly
  without replacement from the set of markets that
  1. have ≥ 100 on-chain trades in the scrape window,
  2. resolve to a token pair in `data/clob_token_map.parquet`, AND
  3. have Gamma metadata (a row in `data/metadata_cache.parquet`).
  - Random seed: 20260424. Documented in `polydata/panel/stratify.py`.
  - Selection excludes any market already picked in the top-100 stratum.

## Measure compute

- Trade-based measures (6): computed with authoritative on-chain trades via
  `trades=load_onchain_trades(...)` injection over the scrape window.
- Quote-based measures: computed over the same scrape window for
  apples-to-apples comparison; full-archive extension is out of scope
  for 3c.
- Missing / insufficient measures report NaN with their existing flags;
  no post-hoc imputation.

## Stylized facts

SF1-SF8 computations follow the design spec §5.2 one-table-per-row:

| SF | Primary metric | Source |
|---|---|---|
| SF1 | Median quoted spread bps per price decile | quote measures |
| SF2 | Distribution of populated L2 levels | quote depth profile |
| SF3 | χ² of quote-update intensity vs Polygon 2s period | quote clock |
| SF4 | Per-market Herfindahl of maker-wallet share | on-chain makers |
| SF5 | Category × measure tables | Gamma category × measures |
| SF6 | p50/p95/p99 of `ts_created − ts_received` | quote latency |
| SF7 | Per-market wash share + recomputed measures w/o wash | on-chain maker==taker |
| SF8 | Depth on log(T−t) regression coefficients | quote depth × Gamma end_date |

## Spread decomposition

Glosten-Harris specification on the top-100 stratum only:
`eff_half_spread = c + φ · sign + ε` where `c` is transitory (realized) and
`φ · sign` is the adverse-selection component. OLS with HC3 SEs; report
per-market c and φ with CIs; aggregate with market fixed effects.

## Commit discipline

- This document is committed BEFORE `scripts/build_panel.py` runs.
- The resulting `data/panel.parquet` hash is written to this doc once
  T2 produces it, creating a one-way dependency.
- Any departure from the rules above (e.g., reducing random-500 to
  random-300 due to insufficient qualifying markets) requires an amendment
  section at the bottom of this file with date + reason.

## Panel hash (T2 output, 2026-04-24)

- File: `data/panel.parquet`
- SHA-256: `eac535cdeae72779ec6aba5022a6e47ea8a8b7295350c17729596ea698dae501`
- Row count: 600 (top stratum 100 + random stratum 500)
- Top-stratum volume range: `$4.56M – $96.0M` USDC
- Random-stratum trade-count range: `100 – 24,378` (all ≥ 100 by construction)
