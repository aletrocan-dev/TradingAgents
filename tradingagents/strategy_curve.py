"""Read a finished sweep's decisions as a position series, against buy & hold.

Deliberately outside ``backtest.py``, whose scope note refuses to grow a
portfolio simulator: nothing here feeds the evaluation engine, and no decision
is made differently because of it. This is one reading of a finished log under
assumptions the caller states out loud — a position rule, a fill convention and
a cost — so the number is only ever as good as those. Whatever renders it prints
them beside it.

What it cannot answer: position sizing, leverage, partial fills, slippage, or
anything about a book holding more than one instrument. A curve is per ticker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradingagents.agents.utils.rating import RATING_REVIEW
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.utils import get_current_date

# Sell-side weights: a rating names a weight to hold, so "Hold" is being in the
# market at neutral weight, not sitting it out. Overridable via config.
DEFAULT_POSITIONS: dict[str, float] = {
    "Buy": 1.0,
    "Overweight": 1.0,
    "Hold": 1.0,
    "Underweight": 0.0,
    "Sell": 0.0,
}


@dataclass
class StrategyCurve:
    """Equity of following the decisions, beside equity of holding the asset."""

    ticker: str
    dates: list[str]
    strategy: list[float]
    buy_hold: list[float]
    positions: list[float] = field(default_factory=list)
    changes: int = 0
    carried: int = 0          # decisions with no readable rating, position kept
    # (bar index, rating) for every decision, so a reader can see where the
    # system acted and what it called rather than inferring it from the line.
    markers: list[tuple[int, str]] = field(default_factory=list)
    cost_bps: float = 0.0
    rule: dict[str, float] = field(default_factory=dict)

    @property
    def strategy_return(self) -> float:
        return self.strategy[-1] - 1.0

    @property
    def buy_hold_return(self) -> float:
        return self.buy_hold[-1] - 1.0

    @property
    def excess(self) -> float:
        return self.strategy_return - self.buy_hold_return

    @property
    def time_in_market(self) -> float:
        return sum(p > 0 for p in self.positions) / len(self.positions) if self.positions else 0.0

    def max_drawdown(self, which: str = "strategy") -> float:
        series = self.strategy if which == "strategy" else self.buy_hold
        peak, worst = series[0], 0.0
        for value in series:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1.0)
        return worst

    def render(self) -> str:
        return (
            f"{self.ticker}: strategy {self.strategy_return:+.1%} vs buy&hold "
            f"{self.buy_hold_return:+.1%} ({self.excess:+.1%} excess, "
            f"{self.time_in_market:.0%} in market, {self.changes} position changes)"
        )


def _target_positions(config: dict) -> dict[str, float]:
    return {**DEFAULT_POSITIONS, **(config.get("strategy_positions") or {})}


def build_curve(entries: list[dict], ticker: str, config: dict) -> StrategyCurve | None:
    """Follow ``ticker``'s decisions bar by bar; ``None`` when it has none.

    A decision dated D is acted on at D's close: the analysis saw data through D,
    so that is the first fill it could have got. A decision with no readable
    rating keeps the position — it is not an order.
    """
    signals = sorted(
        (e for e in entries if e["ticker"] == ticker and e.get("date")),
        key=lambda e: e["date"],
    )
    if not signals:
        return None

    rule = _target_positions(config)
    cost_bps = float(config.get("strategy_cost_bps") or 0.0)

    prices = load_ohlcv(ticker, get_current_date(), fill_gaps=False)
    rows = prices[prices["Date"] >= signals[0]["date"]].reset_index(drop=True)
    if len(rows) < 2:
        return None

    # A signal lands on the first bar at or after its date, so one dated on a day
    # the asset did not trade still takes effect rather than being dropped.
    pending = list(signals)
    by_bar: dict[int, dict] = {}
    for i, bar_date in enumerate(rows["Date"]):
        while pending and pending[0]["date"] <= bar_date.strftime("%Y-%m-%d"):
            by_bar[i] = pending.pop(0)

    curve = StrategyCurve(
        ticker=ticker, dates=[rows["Date"].iloc[0].strftime("%Y-%m-%d")],
        strategy=[1.0], buy_hold=[1.0], cost_bps=cost_bps, rule=rule,
    )
    position = 0.0

    def act(bar: int) -> None:
        nonlocal position
        signal = by_bar.get(bar)
        if signal is None:
            return
        curve.markers.append((bar, signal["rating"]))
        if signal["rating"] == RATING_REVIEW or signal["rating"] not in rule:
            curve.carried += 1
            return
        target = rule[signal["rating"]]
        if target == position:
            return
        curve.strategy[-1] *= 1 - abs(target - position) * cost_bps / 10_000
        curve.changes += 1
        position = target

    act(0)
    closes = rows["Close"].tolist()
    for i in range(1, len(rows)):
        ret = closes[i] / closes[i - 1] - 1.0
        curve.strategy.append(curve.strategy[-1] * (1 + position * ret))
        curve.buy_hold.append(curve.buy_hold[-1] * (1 + ret))
        curve.dates.append(rows["Date"].iloc[i].strftime("%Y-%m-%d"))
        curve.positions.append(position)
        act(i)

    return curve


def build_curves(entries: list[dict], config: dict) -> list[StrategyCurve]:
    """One curve per ticker in the log. Combining them would need portfolio
    weights, which is exactly what this does not model."""
    curves = []
    for ticker in sorted({e["ticker"] for e in entries if e.get("ticker")}):
        try:
            curve = build_curve(entries, ticker, config)
        except Exception:  # a missing price series must not cost the whole report
            continue
        if curve is not None:
            curves.append(curve)
    return curves
