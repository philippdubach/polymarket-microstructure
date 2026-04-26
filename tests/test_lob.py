from decimal import Decimal

from polydata.events import BookSnapshot, PriceChange
from polydata.lob import LOBReplay


def _snap(bids, asks, t=1776066329.0):
    return BookSnapshot(
        update_type="book_snapshot",
        market_id="m1", token_id="t1", side="YES",
        best_bid=Decimal(str(bids[0][0])) if bids else Decimal("0"),
        best_ask=Decimal(str(asks[0][0])) if asks else Decimal("1"),
        timestamp=t,
        bids=[(Decimal(str(p)), Decimal(str(q))) for p, q in bids],
        asks=[(Decimal(str(p)), Decimal(str(q))) for p, q in asks],
    )


def _pc(price, size, side, best_bid, best_ask, t=1776066329.5):
    return PriceChange(
        update_type="price_change",
        market_id="m1", token_id="t1", side="YES",
        best_bid=Decimal(str(best_bid)), best_ask=Decimal(str(best_ask)),
        timestamp=t,
        change_price=Decimal(str(price)), change_size=Decimal(str(size)),
        change_side=side,
    )


def test_snapshot_initialises_book():
    lob = LOBReplay()
    lob.apply(_snap([(0.5, 100), (0.4, 50)], [(0.51, 80), (0.6, 20)]))
    assert lob.best_bid() == Decimal("0.5")
    assert lob.best_ask() == Decimal("0.51")
    assert lob.bid_depth(Decimal("0.5")) == Decimal("100")
    assert lob.ask_depth(Decimal("0.51")) == Decimal("80")


def test_buy_price_change_replaces_level_qty():
    lob = LOBReplay()
    lob.apply(_snap([(0.5, 100)], [(0.51, 80)]))
    lob.apply(_pc(0.5, 60, "BUY", best_bid=0.5, best_ask=0.51))
    assert lob.bid_depth(Decimal("0.5")) == Decimal("60")


def test_zero_size_removes_level():
    lob = LOBReplay()
    lob.apply(_snap([(0.5, 100), (0.4, 50)], [(0.51, 80)]))
    lob.apply(_pc(0.5, 0, "BUY", best_bid=0.4, best_ask=0.51))
    assert lob.bid_depth(Decimal("0.5")) == Decimal("0")
    assert lob.best_bid() == Decimal("0.4")


def test_spread_requires_both_sides():
    lob = LOBReplay()
    assert lob.spread() is None
    lob.apply(_snap([(0.5, 10)], [(0.51, 10)]))
    assert lob.spread() == Decimal("0.01")


def test_top_k_depth():
    lob = LOBReplay()
    lob.apply(_snap([(0.5, 10), (0.49, 5), (0.48, 2)], [(0.51, 8), (0.52, 3)]))
    assert lob.top_k_bid_depth(2) == Decimal("15")
    assert lob.top_k_ask_depth(5) == Decimal("11")
