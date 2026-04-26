from decimal import Decimal

from polydata.classify import TradeSign, classify_trade_direction


def test_above_mid_is_buy():
    assert classify_trade_direction(
        trade_price=Decimal("0.51"),
        prev_best_bid=Decimal("0.49"),
        prev_best_ask=Decimal("0.52"),
        prev_trade_price=None,
    ) is TradeSign.BUY


def test_below_mid_is_sell():
    assert classify_trade_direction(
        trade_price=Decimal("0.49"),
        prev_best_bid=Decimal("0.50"),
        prev_best_ask=Decimal("0.52"),
        prev_trade_price=None,
    ) is TradeSign.SELL


def test_at_mid_uses_tick_rule_up():
    assert classify_trade_direction(
        trade_price=Decimal("0.50"),
        prev_best_bid=Decimal("0.48"),
        prev_best_ask=Decimal("0.52"),
        prev_trade_price=Decimal("0.49"),
    ) is TradeSign.BUY


def test_at_mid_uses_tick_rule_down():
    assert classify_trade_direction(
        trade_price=Decimal("0.50"),
        prev_best_bid=Decimal("0.48"),
        prev_best_ask=Decimal("0.52"),
        prev_trade_price=Decimal("0.51"),
    ) is TradeSign.SELL


def test_at_mid_no_prev_is_unknown():
    assert classify_trade_direction(
        trade_price=Decimal("0.50"),
        prev_best_bid=Decimal("0.48"),
        prev_best_ask=Decimal("0.52"),
        prev_trade_price=None,
    ) is TradeSign.UNKNOWN
