import json
from decimal import Decimal

from polydata.events import BookSnapshot, PriceChange, parse_event


def test_parse_price_change(price_change_raw):
    ev = parse_event(json.dumps(price_change_raw))
    assert isinstance(ev, PriceChange)
    assert ev.side == "NO"
    assert ev.best_bid == Decimal("0.987")
    assert ev.best_ask == Decimal("0.988")
    assert ev.change_side == "BUY"


def test_parse_book_snapshot(book_snapshot_raw):
    ev = parse_event(json.dumps(book_snapshot_raw))
    assert isinstance(ev, BookSnapshot)
    assert ev.side == "YES"
    assert len(ev.bids) == 3
    assert ev.bids[0] == (Decimal("0.001"), Decimal("13390"))
    assert ev.asks[0] == (Decimal("0.999"), Decimal("20029.01"))


def test_rejects_unknown_update_type():
    import pytest

    with pytest.raises(ValueError):
        parse_event(json.dumps({"update_type": "trade"}))
