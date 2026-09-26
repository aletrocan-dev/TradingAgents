"""Following the decisions, against just holding the asset.

The curve is a reading of a finished log under stated assumptions, so the
assumptions are what these tests pin: a decision is acted on at its own close,
a rating names a weight to hold, and a decision nobody can read is not an order.
"""

import pandas as pd
import pytest

from tradingagents.strategy_curve import build_curve, build_curves

# +5%, +4.76%, -4.55%, -4.76%, -5%
CLOSES = [100.0, 105.0, 110.0, 105.0, 100.0, 95.0]
DATES = [f"2026-01-0{d}" for d in range(1, 7)]


@pytest.fixture(autouse=True)
def _prices(monkeypatch):
    frame = pd.DataFrame({"Date": pd.to_datetime(DATES), "Close": CLOSES})
    monkeypatch.setattr(
        "tradingagents.strategy_curve.load_ohlcv", lambda *a, **k: frame.copy()
    )
    # The day after the last bar: a bar dated today is still trading.
    monkeypatch.setattr(
        "tradingagents.strategy_curve.get_current_date", lambda: "2026-01-07"
    )


def _entries(*pairs, ticker="BTC-USD"):
    return [{"ticker": ticker, "date": date, "rating": rating} for date, rating in pairs]


@pytest.mark.unit
def test_holding_throughout_is_exactly_buy_and_hold():
    """A rating names a weight, and Hold is neutral weight: in the market."""
    curve = build_curve(_entries(("2026-01-01", "Hold")), "BTC-USD", {})
    assert curve.strategy == pytest.approx(curve.buy_hold)
    assert curve.buy_hold_return == pytest.approx(-0.05)
    assert curve.excess == pytest.approx(0.0)


@pytest.mark.unit
def test_selling_throughout_never_moves():
    curve = build_curve(_entries(("2026-01-01", "Sell")), "BTC-USD", {})
    assert curve.strategy == [1.0] * len(CLOSES)
    assert curve.strategy_return == pytest.approx(0.0)
    assert curve.time_in_market == 0.0


@pytest.mark.unit
def test_riding_the_rise_and_leaving_before_the_fall():
    """Buy on day 1, Sell on day 3 (the top): +10% held, -5% avoided."""
    curve = build_curve(
        _entries(("2026-01-01", "Buy"), ("2026-01-03", "Sell")), "BTC-USD", {}
    )
    assert curve.strategy_return == pytest.approx(0.10)     # 110/100
    assert curve.buy_hold_return == pytest.approx(-0.05)    # 95/100
    assert curve.excess == pytest.approx(0.15)
    assert curve.positions == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert curve.time_in_market == pytest.approx(0.4)
    assert curve.changes == 2


@pytest.mark.unit
def test_an_unreadable_decision_keeps_the_position():
    """REVIEW is not an order: acting on it would invent a trade nobody called."""
    kept = build_curve(
        _entries(("2026-01-01", "Buy"), ("2026-01-03", "REVIEW")), "BTC-USD", {}
    )
    held = build_curve(_entries(("2026-01-01", "Buy")), "BTC-USD", {})

    assert kept.strategy == pytest.approx(held.strategy)
    assert kept.carried == 1
    assert kept.changes == 1  # only the entry


@pytest.mark.unit
def test_the_decision_is_acted_on_at_its_own_close():
    """The analysis saw data through its date, so that close is the first
    honest fill — the day's own move is not credited to the decision."""
    curve = build_curve(_entries(("2026-01-02", "Buy")), "BTC-USD", {})
    # The curve starts at the signal, so the +5% of 01-01 -> 01-02 is not in it.
    assert curve.dates[0] == "2026-01-02"
    assert curve.strategy_return == pytest.approx(95.0 / 105.0 - 1)


@pytest.mark.unit
def test_costs_are_charged_on_the_turnover():
    free = build_curve(_entries(("2026-01-01", "Buy")), "BTC-USD", {})
    charged = build_curve(
        _entries(("2026-01-01", "Buy")), "BTC-USD", {"strategy_cost_bps": 50}
    )
    assert charged.strategy_return < free.strategy_return
    assert charged.strategy[-1] == pytest.approx(free.strategy[-1] * (1 - 0.005))


@pytest.mark.unit
def test_the_position_rule_is_configurable():
    curve = build_curve(
        _entries(("2026-01-01", "Hold")), "BTC-USD", {"strategy_positions": {"Hold": 0.0}}
    )
    assert curve.strategy_return == pytest.approx(0.0)


@pytest.mark.unit
def test_a_ticker_with_no_decisions_has_no_curve():
    assert build_curve(_entries(("2026-01-01", "Buy"), ticker="NVDA"), "BTC-USD", {}) is None


@pytest.mark.unit
def test_one_curve_per_ticker():
    entries = _entries(("2026-01-01", "Buy")) + _entries(("2026-01-01", "Sell"), ticker="NVDA")
    curves = build_curves(entries, {})
    assert [c.ticker for c in curves] == ["BTC-USD", "NVDA"]


@pytest.mark.unit
def test_drawdown_is_measured_on_each_series():
    curve = build_curve(_entries(("2026-01-01", "Hold")), "BTC-USD", {})
    # 110 -> 95 from the peak
    assert curve.max_drawdown("buy_hold") == pytest.approx(95.0 / 110.0 - 1)


@pytest.mark.unit
def test_the_summary_line_names_both_sides():
    line = build_curve(_entries(("2026-01-01", "Buy"), ("2026-01-03", "Sell")), "BTC-USD", {}).render()
    assert "strategy +10.0%" in line and "buy&hold -5.0%" in line


@pytest.mark.unit
def test_the_curve_ends_where_the_last_decision_is_judged():
    """Holding one day, the last call (01-03) is judged at 01-04's close: the
    days after it belong to no decision and must not move either line."""
    curve = build_curve(_entries(("2026-01-01", "Buy"), ("2026-01-03", "Hold")), "BTC-USD",
                        {"holding_period_days": 1})
    assert curve.dates[-1] == "2026-01-04"
    assert curve.buy_hold_return == pytest.approx(0.05)


@pytest.mark.unit
def test_when_the_report_is_written_does_not_change_the_curve(monkeypatch):
    """Read the day after the sweep or a week later, the same decisions give
    the same numbers: that is what makes two sweeps' reports comparable."""
    config = {"holding_period_days": 1}
    entries = _entries(("2026-01-01", "Buy"), ("2026-01-02", "Sell"))
    early = build_curve(entries, "BTC-USD", config)
    monkeypatch.setattr("tradingagents.strategy_curve.get_current_date", lambda: "2026-03-01")
    later = build_curve(entries, "BTC-USD", config)
    assert (early.dates, early.strategy, early.buy_hold) == (later.dates, later.strategy, later.buy_hold)


@pytest.mark.unit
def test_a_bar_still_trading_today_is_left_out(monkeypatch):
    monkeypatch.setattr("tradingagents.strategy_curve.get_current_date", lambda: "2026-01-06")
    curve = build_curve(_entries(("2026-01-01", "Hold")), "BTC-USD", {})
    assert curve.dates[-1] == "2026-01-05"


# --- decisions read as orders on the portfolio -------------------------------
# CLOSES = 100, 105, 110, 105, 100, 95 on 01-01..01-06. Expected values are
# worked out by hand from cash and coins held, not from the code under test.

ORDERS = {"Buy": 1.0, "Overweight": 0.5, "Underweight": -0.5, "Sell": -1.0}


def _orders(*pairs):
    return build_curve(_entries(*pairs), "BTC-USD", {"strategy_trades": ORDERS})


@pytest.mark.unit
def test_buying_with_everything_is_buy_and_hold():
    curve = _orders(("2026-01-01", "Buy"))
    assert curve.strategy == pytest.approx(curve.buy_hold)


@pytest.mark.unit
def test_half_the_portfolio_bought_is_held_as_coins_not_rebalanced():
    """0.5 in cash and 0.5 in coins bought at 100: at 95 that is
    0.5 + 0.5 * 0.95 = 0.975. Keeping a constant 50% weight would give
    another number; nobody holding coins rebalances every day."""
    curve = _orders(("2026-01-01", "Overweight"))
    assert curve.strategy[-1] == pytest.approx(0.975)
    assert curve.positions[0] == pytest.approx(0.5)
    assert curve.positions[1] == pytest.approx(0.5 * 1.05 / 1.025)  # drifted with the price


@pytest.mark.unit
def test_selling_half_the_portfolio_sells_half_its_value():
    """All in at 100; at 110 the portfolio is 1.10 and Underweight sells 0.55
    of it, leaving 0.55 in coins that are worth 0.55 * 95 / 110 at the end."""
    curve = _orders(("2026-01-01", "Buy"), ("2026-01-03", "Underweight"))
    assert curve.strategy[-1] == pytest.approx(0.55 + 0.55 * 95 / 110)
    assert curve.final_position == pytest.approx(0.55 * 95 / 110 / (0.55 + 0.55 * 95 / 110))


@pytest.mark.unit
def test_there_is_no_short_and_no_leverage():
    out = _orders(("2026-01-01", "Sell"), ("2026-01-02", "Underweight"))
    assert out.strategy == pytest.approx([1.0] * 6) and out.changes == 0
    full = _orders(("2026-01-01", "Overweight"), ("2026-01-02", "Overweight"),
                   ("2026-01-03", "Buy"))
    assert full.changes == 2                      # the third order finds no cash left
    assert max(full.positions) <= 1.0 + 1e-12


@pytest.mark.unit
def test_hold_trades_nothing_and_review_is_counted_as_unreadable():
    curve = _orders(("2026-01-01", "Overweight"), ("2026-01-02", "Hold"), ("2026-01-03", "REVIEW"))
    assert curve.strategy[-1] == pytest.approx(0.975)
    assert (curve.changes, curve.carried) == (1, 1)


@pytest.mark.unit
@pytest.mark.parametrize("text, expected", [
    ("Buy=1,Overweight=0.5,Underweight=-0.5,Sell=-1", ORDERS),
    ("buy=100%, sell=-100%", {"Buy": 1.0, "Sell": -1.0}),
])
def test_an_order_rule_reads_from_one_line(text, expected):
    from tradingagents.strategy_curve import parse_trades

    assert parse_trades(text) == expected


@pytest.mark.unit
@pytest.mark.parametrize("text", ["Moon=1", "Buy", "Buy=two", "Buy=1.5", ""])
def test_an_order_rule_that_makes_no_sense_is_refused(text):
    from tradingagents.strategy_curve import parse_trades

    with pytest.raises(ValueError):
        parse_trades(text)
