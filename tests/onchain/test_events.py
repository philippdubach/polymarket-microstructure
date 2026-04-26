from polydata.onchain.events import (
    ORDER_FILLED_TOPIC,
    OrderFilledEvent,
    decode_order_filled,
)


def test_topic_hash_length():
    assert isinstance(ORDER_FILLED_TOPIC, str)
    assert ORDER_FILLED_TOPIC.startswith("0x")
    assert len(ORDER_FILLED_TOPIC) == 66


def test_decode_sample_log():
    data = (
        "0x"
        + "0000000000000000000000000000000000000000000000000000000000000001"
        + "0000000000000000000000000000000000000000000000000000000000000002"
        + "00000000000000000000000000000000000000000000000000000000000003e8"
        + "00000000000000000000000000000000000000000000000000000000000001f4"
        + "0000000000000000000000000000000000000000000000000000000000000001"
    )
    log = {
        "blockNumber": "0x10",
        "transactionHash": "0xdeadbeef",
        "logIndex": "0x0",
        "topics": [
            ORDER_FILLED_TOPIC,
            "0x" + "aa" * 32,
            "0x" + "00" * 12 + "bb" * 20,
            "0x" + "00" * 12 + "cc" * 20,
        ],
        "data": data,
    }
    ev = decode_order_filled(log)
    assert isinstance(ev, OrderFilledEvent)
    assert ev.block_number == 0x10
    assert ev.tx_hash == "0xdeadbeef"
    assert ev.log_index == 0
    assert ev.maker == "0x" + "bb" * 20
    assert ev.taker == "0x" + "cc" * 20
    assert ev.maker_asset_id == 1
    assert ev.taker_asset_id == 2
    assert ev.maker_amount_filled == 1000
    assert ev.taker_amount_filled == 500
    assert ev.fee == 1
