from __future__ import annotations

from decimal import Decimal

from polydata.events import BookSnapshot, Event, PriceChange


class LOBReplay:
    """Single-side LOB state machine.

    Applies Polymarket price_change and book_snapshot events in sequence.
    A price_change with change_size=0 removes the level. Positive sizes
    replace the level's resting quantity (Polymarket semantics, verified
    against public snapshots).
    """

    def __init__(self) -> None:
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._best_bid_hint: Decimal | None = None
        self._best_ask_hint: Decimal | None = None

    def apply(self, event: Event) -> None:
        if isinstance(event, BookSnapshot):
            self._bids = {p: q for p, q in event.bids if q > 0}
            self._asks = {p: q for p, q in event.asks if q > 0}
            self._best_bid_hint = event.best_bid
            self._best_ask_hint = event.best_ask
        elif isinstance(event, PriceChange):
            side_book = self._bids if event.change_side == "BUY" else self._asks
            p = event.change_price
            q = event.change_size
            if q == 0:
                side_book.pop(p, None)
            else:
                side_book[p] = q
            self._best_bid_hint = event.best_bid
            self._best_ask_hint = event.best_ask

    def best_bid(self) -> Decimal | None:
        if not self._bids:
            return None
        return max(self._bids)

    def best_ask(self) -> Decimal | None:
        if not self._asks:
            return None
        return min(self._asks)

    def spread(self) -> Decimal | None:
        b = self.best_bid()
        a = self.best_ask()
        if b is None or a is None:
            return None
        return a - b

    def mid(self) -> Decimal | None:
        b = self.best_bid()
        a = self.best_ask()
        if b is None or a is None:
            return None
        return (a + b) / 2

    def bid_depth(self, price: Decimal) -> Decimal:
        return self._bids.get(price, Decimal("0"))

    def ask_depth(self, price: Decimal) -> Decimal:
        return self._asks.get(price, Decimal("0"))

    def top_k_bid_depth(self, k: int) -> Decimal:
        levels = sorted(self._bids.items(), key=lambda kv: -kv[0])[:k]
        return sum((q for _, q in levels), Decimal("0"))

    def top_k_ask_depth(self, k: int) -> Decimal:
        levels = sorted(self._asks.items(), key=lambda kv: kv[0])[:k]
        return sum((q for _, q in levels), Decimal("0"))

    def snapshot(self) -> dict[str, list[tuple[Decimal, Decimal]]]:
        return {
            "bids": sorted(self._bids.items(), key=lambda kv: -kv[0]),
            "asks": sorted(self._asks.items(), key=lambda kv: kv[0]),
        }
