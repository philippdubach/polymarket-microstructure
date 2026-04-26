"""On-chain data ingestion + calibration."""

from polydata.onchain.events import (
    ORDER_FILLED_TOPIC,
    OrderFilledEvent,
    decode_order_filled,
)
from polydata.onchain.join import (
    JoinResult,
    aggregate_onchain_by_block_tx,
    aggregate_onchain_df,
    block_level_match,
    compute_metrics,
    decode_fill_to_trade,
)
from polydata.onchain.rpc import PolygonRpcClient
from polydata.onchain.scraper import (
    FINALITY_BUFFER_BLOCKS,
    scrape_order_filled,
    scrape_slice_finalized,
)
from polydata.onchain.token_map import (
    TOKEN_MAP_CACHE,
    TOKEN_MAP_SCHEMA,
    fetch_clob_token_ids_bulk,
    load_token_map,
    parse_gamma_clob_token_ids,
    resolve_token_ids,
    save_token_map,
)
from polydata.onchain.trades import (
    ONCHAIN_TRADE_SCHEMA,
    OnchainTrade,
    OnchainTradeStream,
    attach_pre_trade_mid,
    load_onchain_trades,
)

__all__ = [
    "FINALITY_BUFFER_BLOCKS",
    "JoinResult",
    "ONCHAIN_TRADE_SCHEMA",
    "ORDER_FILLED_TOPIC",
    "OnchainTrade",
    "OnchainTradeStream",
    "OrderFilledEvent",
    "PolygonRpcClient",
    "TOKEN_MAP_CACHE",
    "TOKEN_MAP_SCHEMA",
    "aggregate_onchain_by_block_tx",
    "aggregate_onchain_df",
    "attach_pre_trade_mid",
    "block_level_match",
    "compute_metrics",
    "decode_fill_to_trade",
    "decode_order_filled",
    "fetch_clob_token_ids_bulk",
    "load_onchain_trades",
    "load_token_map",
    "parse_gamma_clob_token_ids",
    "resolve_token_ids",
    "save_token_map",
    "scrape_order_filled",
    "scrape_slice_finalized",
]
