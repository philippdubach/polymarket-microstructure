import pytest

PRICE_CHANGE_RAW = {
    "update_type": "price_change",
    "market_id": "0x00000977017fa72fb6b1908ae694000d3b51f442c2552656b10bdbbfd16ff707",
    "token_id": "56914066788195652124819518509742797268065105214859208750975274268335607133892",
    "side": "NO",
    "best_bid": "0.987",
    "best_ask": "0.988",
    "timestamp": 1776241838.000965,
    "change_price": "0.808",
    "change_size": "5",
    "change_side": "BUY",
}

BOOK_SNAPSHOT_RAW = {
    "update_type": "book_snapshot",
    "market_id": "0x0008043c3ed513ecff7ee64380fc943dc73eb3dfb6674f281149efe4769f7515",
    "token_id": "97684905927345553455494278582909124912046930226695064344571162061840768197777",
    "side": "YES",
    "best_bid": "0.03",
    "best_ask": "0.031",
    "timestamp": 1776066329.4319417,
    "bids": [["0.001", "13390"], ["0.002", "877.63"], ["0.03", "24.36"]],
    "asks": [["0.999", "20029.01"], ["0.998", "67.75"], ["0.95", "20"]],
}


@pytest.fixture
def price_change_raw():
    return PRICE_CHANGE_RAW


@pytest.fixture
def book_snapshot_raw():
    return BOOK_SNAPSHOT_RAW
