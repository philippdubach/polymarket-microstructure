# polydata/measures/trades.py
"""Trade inference from orderbook events (strict + loose variants).

LOOSE: every size decrement is signed as a trade (known 40-70% FP rate).
STRICT: top-of-book decrements co-timed (within co_timing_ms) with an
opposite-side book change on the same token, aggregated to block_seconds.

Sign convention: change_side == "SELL" (ask hit) → +1 (buyer-initiated);
change_side == "BUY" (bid hit) → -1 (seller-initiated). The feed's
change_side labels the RESTING side, not the aggressor — aggressor sign
is the negation.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import polars as pl

from polydata.events import BookSnapshot, PriceChange
from polydata.window import MeasurementWindow

_RESTING_SIDE_TO_AGGRESSOR_SIGN: dict[str, int] = {"SELL": 1, "BUY": -1}

_TRADE_SCHEMA: dict = {
    "market_id": pl.Utf8,
    "t": pl.Datetime(time_zone="UTC"),
    "token_id": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "sign": pl.Int8,
}


def infer_trades_loose(window: MeasurementWindow) -> pl.DataFrame:
    rows: list[dict] = []
    last_size: dict[tuple[str, str, Decimal], Decimal] = {}
    for rec in window.events:
        ev = rec.event
        if isinstance(ev, BookSnapshot):
            for p, q in ev.bids:
                last_size[(ev.token_id, "BUY", p)] = q
            for p, q in ev.asks:
                last_size[(ev.token_id, "SELL", p)] = q
            continue
        if not isinstance(ev, PriceChange):
            continue
        key = (ev.token_id, ev.change_side, ev.change_price)
        prev = last_size.get(key, Decimal("0"))
        new = ev.change_size
        if new < prev:
            decrement = prev - new
            sign = _RESTING_SIDE_TO_AGGRESSOR_SIGN.get(ev.change_side, 0)
            if sign != 0 and decrement > 0:
                rows.append({
                    "market_id": window.market_id,
                    "t": rec.ts_received,
                    "token_id": ev.token_id,
                    "price": float(ev.change_price),
                    "size": float(decrement),
                    "sign": int(sign),
                })
        last_size[key] = new
    return pl.DataFrame(rows, schema=_TRADE_SCHEMA) if rows else pl.DataFrame(schema=_TRADE_SCHEMA)


def infer_trades_strict(
    window: MeasurementWindow,
    co_timing_ms: int = 100,
    block_seconds: int = 2,
    top_n_levels: int = 1,
) -> pl.DataFrame:
    last_size: dict[tuple[str, str, Decimal], Decimal] = {}
    best_bid: dict[str, Decimal | None] = {}
    best_ask: dict[str, Decimal | None] = {}
    # recent: past PriceChange events within co_timing_ms window
    recent: list[tuple] = []
    # pending: top-of-book decrements awaiting a future companion
    # list of (deadline_ts, candidate_dict, opposite_side, token_id)
    pending: list[tuple] = []
    candidates: list[dict] = []
    window_delta = timedelta(milliseconds=co_timing_ms)

    def _top_n_prices(token_id: str, change_side: str) -> set[Decimal]:
        subset = [
            p for (t, s, p), q in last_size.items()
            if t == token_id and s == change_side and q > 0
        ]
        if not subset:
            return set()
        if change_side == "BUY":
            subset.sort(reverse=True)
        else:
            subset.sort()
        return set(subset[:top_n_levels])

    for rec in window.events:
        ev = rec.event
        if isinstance(ev, BookSnapshot):
            for p, q in ev.bids:
                last_size[(ev.token_id, "BUY", p)] = q
            for p, q in ev.asks:
                last_size[(ev.token_id, "SELL", p)] = q
            best_bid[ev.token_id] = ev.best_bid
            best_ask[ev.token_id] = ev.best_ask
            continue
        if not isinstance(ev, PriceChange):
            continue

        cutoff = rec.ts_received - window_delta
        recent = [(ts, e) for ts, e in recent if ts >= cutoff]

        # Check if this PriceChange resolves any pending decrements
        new_pending: list[tuple] = []
        for deadline, cand, opp_side, tok in pending:
            if rec.ts_received <= deadline:
                if (
                    isinstance(ev, PriceChange)
                    and ev.token_id == tok
                    and ev.change_side == opp_side
                ):
                    candidates.append(cand)
                else:
                    new_pending.append((deadline, cand, opp_side, tok))
            # expired pending entries are simply dropped
        pending = new_pending

        key = (ev.token_id, ev.change_side, ev.change_price)
        prev = last_size.get(key, Decimal("0"))
        new = ev.change_size
        is_decrement = new < prev
        is_top_n = ev.change_price in _top_n_prices(ev.token_id, ev.change_side)

        if is_decrement and is_top_n:
            opposite = "BUY" if ev.change_side == "SELL" else "SELL"
            # Check past companions first
            has_past_companion = any(
                isinstance(e, PriceChange)
                and e.token_id == ev.token_id
                and e.change_side == opposite
                for _, e in recent
            )
            sign = _RESTING_SIDE_TO_AGGRESSOR_SIGN[ev.change_side]
            cand = {
                "market_id": window.market_id,
                "t": rec.ts_received,
                "token_id": ev.token_id,
                "price": float(ev.change_price),
                "size": float(prev - new),
                "sign": int(sign),
            }
            if has_past_companion:
                candidates.append(cand)
            else:
                # Queue for future companion within co_timing_ms
                deadline = rec.ts_received + window_delta
                pending.append((deadline, cand, opposite, ev.token_id))

        last_size[key] = new
        if ev.best_bid is not None:
            best_bid[ev.token_id] = ev.best_bid
        if ev.best_ask is not None:
            best_ask[ev.token_id] = ev.best_ask
        recent.append((rec.ts_received, ev))

    raw = (
        pl.DataFrame(candidates, schema=_TRADE_SCHEMA)
        if candidates
        else pl.DataFrame(schema=_TRADE_SCHEMA)
    )
    return aggregate_trades_to_blocks(raw, block_seconds=block_seconds)


def aggregate_trades_to_blocks(
    trades: pl.DataFrame,
    block_seconds: int = 2,
) -> pl.DataFrame:
    if trades.height == 0:
        return trades
    bucket = timedelta(seconds=block_seconds)
    return (
        trades
        .with_columns(pl.col("t").dt.truncate(bucket).alias("block_t"))
        .group_by(["market_id", "block_t", "token_id", "price", "sign"])
        .agg(pl.col("size").sum())
        .rename({"block_t": "t"})
        .select(["market_id", "t", "token_id", "price", "size", "sign"])
        .sort("t")
        .with_columns([pl.col(k).cast(v) for k, v in _TRADE_SCHEMA.items()])
    )
