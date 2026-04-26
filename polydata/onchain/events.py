"""OrderFilled ABI + decoder."""

from __future__ import annotations

from dataclasses import dataclass

from eth_abi import decode
from eth_utils import keccak, to_hex

ORDER_FILLED_SIGNATURE = (
    "OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
)
ORDER_FILLED_TOPIC = to_hex(keccak(text=ORDER_FILLED_SIGNATURE))


@dataclass(frozen=True)
class OrderFilledEvent:
    block_number: int
    tx_hash: str
    log_index: int
    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount_filled: int
    taker_amount_filled: int
    fee: int


def _topic_to_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def decode_order_filled(log: dict) -> OrderFilledEvent:
    topics = log["topics"]
    data_hex = log["data"]
    data_bytes = (
        bytes.fromhex(data_hex[2:]) if data_hex.startswith("0x") else bytes.fromhex(data_hex)
    )
    (
        maker_asset_id,
        taker_asset_id,
        maker_amount_filled,
        taker_amount_filled,
        fee,
    ) = decode(
        ["uint256", "uint256", "uint256", "uint256", "uint256"],
        data_bytes,
    )
    return OrderFilledEvent(
        block_number=int(log["blockNumber"], 16),
        tx_hash=log["transactionHash"],
        log_index=int(log["logIndex"], 16),
        order_hash=topics[1],
        maker=_topic_to_address(topics[2]),
        taker=_topic_to_address(topics[3]),
        maker_asset_id=int(maker_asset_id),
        taker_asset_id=int(taker_asset_id),
        maker_amount_filled=int(maker_amount_filled),
        taker_amount_filled=int(taker_amount_filled),
        fee=int(fee),
    )
