from __future__ import annotations

from decimal import Decimal
from enum import Enum


class TradeSign(Enum):
    BUY = 1
    SELL = -1
    UNKNOWN = 0


def classify_trade_direction(
    trade_price: Decimal,
    prev_best_bid: Decimal,
    prev_best_ask: Decimal,
    prev_trade_price: Decimal | None,
) -> TradeSign:
    """Lee-Ready (1991) classifier.

    Quote rule: trade above mid → BUY, below mid → SELL.
    Tick rule (at-mid fallback): compare to prev trade price.
    """
    mid = (prev_best_bid + prev_best_ask) / 2
    if trade_price > mid:
        return TradeSign.BUY
    if trade_price < mid:
        return TradeSign.SELL
    if prev_trade_price is None:
        return TradeSign.UNKNOWN
    if trade_price > prev_trade_price:
        return TradeSign.BUY
    if trade_price < prev_trade_price:
        return TradeSign.SELL
    return TradeSign.UNKNOWN
