from __future__ import annotations

import json
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Side = Literal["YES", "NO"]
TradeSide = Literal["BUY", "SELL"]


def _to_decimal(v: object) -> Decimal:
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))


def _to_decimal_opt(v: object) -> Decimal | None:
    if v is None:
        return None
    return _to_decimal(v)


def _parse_level_list(raw: list[list[object]]) -> list[tuple[Decimal, Decimal]]:
    return [(_to_decimal(p), _to_decimal(q)) for p, q in raw]


class PriceChange(BaseModel):
    update_type: Literal["price_change"]
    market_id: str
    token_id: str
    side: Side
    best_bid: Decimal
    best_ask: Decimal
    timestamp: float
    change_price: Decimal
    change_size: Decimal
    change_side: TradeSide

    @field_validator("best_bid", "best_ask", "change_price", "change_size", mode="before")
    @classmethod
    def _dec(cls, v):  # noqa: D401
        return _to_decimal(v)


class BookSnapshot(BaseModel):
    update_type: Literal["book_snapshot"]
    market_id: str
    token_id: str
    side: Side
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    timestamp: float
    bids: list[tuple[Decimal, Decimal]] = Field(default_factory=list)
    asks: list[tuple[Decimal, Decimal]] = Field(default_factory=list)

    @field_validator("best_bid", "best_ask", mode="before")
    @classmethod
    def _dec(cls, v):
        return _to_decimal_opt(v)

    @field_validator("bids", "asks", mode="before")
    @classmethod
    def _lvl(cls, v):
        return _parse_level_list(v)


Event = PriceChange | BookSnapshot


def parse_event(raw_json: str) -> Event:
    d = json.loads(raw_json)
    t = d.get("update_type")
    if t == "price_change":
        return PriceChange.model_validate(d)
    if t == "book_snapshot":
        return BookSnapshot.model_validate(d)
    raise ValueError(f"unknown update_type: {t!r}")
