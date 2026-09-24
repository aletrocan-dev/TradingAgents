"""The strategy curve drawn in the terminal while a sweep runs.

A sweep is silent for hours while the graph runs. The chart is what it shows
after each cell, so these tests pin what a reader relies on: both lines are
there and tell apart without colour, it fits the width it was given, and the
call under each column is the one that set the position shown beside it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cli.curve_chart import render_curve_chart
from tradingagents.strategy_curve import build_curve

HEAVY, LIGHT = set("━┃┏┓┗┛"), set("─│╭╮╰╯")


def _curve(monkeypatch, closes, ratings, until=None):
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    frame = pd.DataFrame({"Date": dates, "Close": closes})
    monkeypatch.setattr("tradingagents.strategy_curve.load_ohlcv", lambda *a, **k: frame.copy())
    monkeypatch.setattr("tradingagents.strategy_curve.get_current_date",
                        lambda: dates[-1].strftime("%Y-%m-%d"))
    entries = [{"ticker": "BTC-USD", "date": d.strftime("%Y-%m-%d"), "rating": r}
               for d, r in zip(dates, ratings, strict=False) if r]
    return build_curve(entries, "BTC-USD", {}, until=until)


def _plain(curve, width=80):
    return render_curve_chart(curve, width=width).plain


def _row(chart: str, label: str) -> str:
    line = next(line for line in chart.splitlines() if line.lstrip().startswith(label))
    return line[len("in mkt") + 2:]


@pytest.mark.unit
def test_both_lines_are_drawn_and_differ_in_stroke(monkeypatch):
    curve = _curve(monkeypatch, [100, 110, 99, 120, 130], ["Buy", "Sell", None, "Buy", None])
    chart = _plain(curve)
    assert HEAVY & set(chart)   # the strategy
    assert LIGHT & set(chart)   # buy & hold, where the strategy is not on top of it
    assert "strategy" in chart and "buy&hold" in chart
    assert "2026-01-01" in chart and "2026-01-05" in chart


@pytest.mark.unit
def test_the_header_states_the_returns_being_compared(monkeypatch):
    curve = _curve(monkeypatch, [100, 110, 121], ["Hold", None, None])
    chart = _plain(curve)
    assert "strategy +21.0%" in chart and "buy&hold +21.0%" in chart
    assert "excess +0.0 pp" in chart


@pytest.mark.unit
@pytest.mark.parametrize("width", [60, 80, 110])
def test_it_fits_the_width_it_was_given(monkeypatch, width):
    closes = [100 + (i % 7) * 3 - i * 0.2 for i in range(300)]  # more bars than columns
    ratings = ["Buy" if i % 5 else "Sell" for i in range(300)]
    chart = _plain(_curve(monkeypatch, closes, ratings), width=width)
    body = chart.splitlines()[2:-1]  # the header and legend are prose, not layout
    assert max(len(line) for line in body) <= width


@pytest.mark.unit
def test_each_call_sits_over_the_position_it_set(monkeypatch):
    """Out after a Sell, in after a Buy, in the same column as the letter."""
    curve = _curve(monkeypatch, [100, 101, 102, 103, 104],
                   ["Buy", "Sell", "Sell", "Buy", "Underweight"])
    chart = _plain(curve)
    calls, held = _row(chart, "calls"), _row(chart, "in mkt")
    for col, letter in enumerate(calls):
        if letter in "BH":
            assert held[col] == "▀", (col, calls, held)
        elif letter in "SU":
            assert held[col] == "·", (col, calls, held)
    assert [c for c in calls if c != " "] == ["B", "S", "S", "B", "U"]


@pytest.mark.unit
def test_the_latest_call_is_shown_even_when_it_moved_nothing_yet(monkeypatch):
    """In a running sweep the last decision is the one just taken."""
    curve = _curve(monkeypatch, [100, 101, 102], ["Buy", None, "Sell"])
    calls = _row(_plain(curve), "calls").rstrip()
    assert calls.endswith("S")


@pytest.mark.unit
def test_an_unreadable_decision_is_marked_for_review(monkeypatch):
    curve = _curve(monkeypatch, [100, 101, 102], ["Buy", "REVIEW", None])
    assert "?" in _row(_plain(curve), "calls")


@pytest.mark.unit
def test_a_flat_market_draws_without_dividing_by_zero(monkeypatch):
    curve = _curve(monkeypatch, [100, 100, 100], ["Buy", None, None])
    assert "0%" in _plain(curve)


@pytest.mark.unit
def test_a_curve_can_stop_at_the_cell_a_sweep_has_reached(monkeypatch):
    curve = _curve(monkeypatch, [100, 110, 121, 50], ["Buy", "Buy", "Buy", "Buy"],
                   until="2026-01-03")
    assert curve.dates[-1] == "2026-01-03"
    assert curve.buy_hold_return == pytest.approx(0.21)
    assert len(curve.markers) == 3
    assert curve.final_position == 1.0
