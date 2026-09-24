"""A sweep's strategy curve, drawn in the terminal.

The HTML report is the full reading. This is the glance a sweep that runs for
hours needs while it runs: how following the decisions has done against just
holding the asset, day by day, and when the strategy was in the market.

Both lines share one axis of cumulative return from the first decision. They
differ in stroke weight, not only in colour — heavy for the strategy, light for
buy & hold — so the chart reads the same in a terminal without colour or in a
log file.
"""

from __future__ import annotations

from rich.text import Text

from tradingagents.agents.utils.rating import RATING_REVIEW
from tradingagents.strategy_curve import StrategyCurve

# flat, vertical, and the four corners of a step: leaving a row upward/downward
# on the left of a column, arriving at the new row on its right.
_LIGHT = {"flat": "─", "vert": "│", "rise_from": "╯", "rise_to": "╭", "fall_from": "╮", "fall_to": "╰"}
_HEAVY = {"flat": "━", "vert": "┃", "rise_from": "┛", "rise_to": "┏", "fall_from": "┓", "fall_to": "┗"}

_STRATEGY_STYLE = "bold bright_cyan"
_HOLD_STYLE = "yellow"
_AXIS_STYLE = "grey50"

_LETTER = {"Buy": "B", "Overweight": "O", "Hold": "H", "Underweight": "U", "Sell": "S"}
_LETTER_STYLE = {"B": "green", "O": "green", "H": "grey62", "U": "red", "S": "red", "?": "yellow"}


def _sample(n_points: int, columns: int) -> list[int]:
    """Bar index shown at each of ``columns + 1`` chart points.

    Downsampling keeps the nearest bar; with fewer bars than columns a bar
    repeats, which draws as a step on the day it changed rather than a slope
    between two days that never happened.
    """
    return [round(c * (n_points - 1) / columns) for c in range(columns + 1)]


def _draw(grid: list[list[tuple[str, str]]], rows: list[int], glyphs: dict, style: str) -> None:
    height = len(grid)
    for col, (y0, y1) in enumerate(zip(rows, rows[1:], strict=False)):
        if y0 == y1:
            grid[height - 1 - y0][col] = (glyphs["flat"], style)
            continue
        rising = y1 > y0
        grid[height - 1 - y0][col] = (glyphs["rise_from" if rising else "fall_from"], style)
        grid[height - 1 - y1][col] = (glyphs["rise_to" if rising else "fall_to"], style)
        for y in range(min(y0, y1) + 1, max(y0, y1)):
            grid[height - 1 - y][col] = (glyphs["vert"], style)


def render_curve_chart(curve: StrategyCurve, width: int = 88, height: int = 12) -> Text:
    """The curve as a block of terminal text, ``width`` characters wide."""
    n = len(curve.dates)
    strategy = [v - 1.0 for v in curve.strategy]
    hold = [v - 1.0 for v in curve.buy_hold]
    lo = min(0.0, *strategy, *hold)
    hi = max(0.0, *strategy, *hold)
    if hi - lo < 1e-9:
        hi = lo + 0.01

    def row(value: float) -> int:
        return round((value - lo) / (hi - lo) * (height - 1))

    labels = {row(hi): f"{hi:+.1%}", row(lo): f"{lo:+.1%}", row(0.0): "0%"}
    label_width = max(len("in mkt"), *(len(label) for label in labels.values()))
    # label, space, axis, the plot, and one column more for the last bar's call.
    columns = max(10, width - label_width - 3)
    picks = _sample(n, columns)

    grid = [[(" ", "")] * columns for _ in range(height)]
    zero = height - 1 - row(0.0)
    grid[zero] = [("┈", _AXIS_STYLE)] * columns
    _draw(grid, [row(hold[i]) for i in picks], _LIGHT, _HOLD_STYLE)
    _draw(grid, [row(strategy[i]) for i in picks], _HEAVY, _STRATEGY_STYLE)

    out = Text()
    out.append(f"{curve.ticker}  {curve.dates[0]} → {curve.dates[-1]}", style="bold")
    out.append(f"  ·  {curve.time_in_market:.0%} in market, {curve.changes} changes\n")
    out.append(f"strategy {curve.strategy_return:+.1%}", style=_STRATEGY_STYLE)
    out.append("   ")
    out.append(f"buy&hold {curve.buy_hold_return:+.1%}", style=_HOLD_STYLE)
    out.append(f"   excess {curve.excess * 100:+.1f} pp\n", style="bold")

    for r, cells in enumerate(grid):
        y = height - 1 - r
        label = labels.get(y, "")
        out.append(f"{label:>{label_width}} ", style=_AXIS_STYLE)
        out.append("┼" if r == zero else "┤" if label else "│", style=_AXIS_STYLE)
        for char, style in cells:
            out.append(char, style=style)
        out.append("\n")

    pad = " " * (label_width + 1)
    out.append(f"{pad}└{'─' * columns}\n", style=_AXIS_STYLE)
    first, last = curve.dates[0], curve.dates[-1]
    middle = curve.dates[picks[columns // 2]]
    if columns >= 3 * len(first) + 4:
        gap = columns - 3 * len(first)
        axis = first + " " * (gap // 2) + middle + " " * (gap - gap // 2) + last
    else:
        axis = first + " " * max(1, columns - 2 * len(first)) + last
    out.append(f"{pad} {axis}\n", style=_AXIS_STYLE)

    # Column c draws the move from bar picks[c] onward. Under it: the latest
    # decision taken up to that bar, and what it left the strategy holding.
    # One column more than the plot, for the last bar: in a sweep still running
    # that is the decision just taken, the one worth seeing.
    in_market, decided = Text(), Text()
    decisions = dict(curve.markers)
    for c in range(columns + 1):
        bar = picks[c]
        held = curve.positions[bar] if bar < len(curve.positions) else curve.final_position
        if held >= 1.0:
            in_market.append("▀", style="green")
        elif held > 0.0:
            in_market.append("▄", style="green")
        else:
            in_market.append("·", style=_AXIS_STYLE)
        start = picks[c - 1] + 1 if c else 0
        calls = [decisions[b] for b in range(start, bar + 1) if b in decisions]
        if calls:
            letter = "?" if calls[-1] == RATING_REVIEW else _LETTER.get(calls[-1], "?")
            decided.append(letter, style=_LETTER_STYLE[letter])
        else:
            decided.append(" ")
    out.append(f"{'in mkt':>{label_width}}  ", style=_AXIS_STYLE)
    out.append_text(in_market)
    out.append("\n")
    out.append(f"{'calls':>{label_width}}  ", style=_AXIS_STYLE)
    out.append_text(decided)
    out.append("\n")

    out.append(f"{pad} ")
    out.append("━ strategy", style=_STRATEGY_STYLE)
    out.append("  ")
    out.append("─ buy&hold", style=_HOLD_STYLE)
    out.append("  ▀ in market  calls: B O H U S ?=review", style=_AXIS_STYLE)
    return out
