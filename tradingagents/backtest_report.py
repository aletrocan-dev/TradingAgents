"""Self-contained HTML report for a backtest run.

Single file, no external assets: inline CSS and hand-rolled inline SVG charts
(stdlib only — the project ships no plotting/HTML dependency and this keeps it
that way), so the report opens and reads correctly with no network access.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from tradingagents.agents.utils.rating import RATING_REVIEW

if TYPE_CHECKING:
    from tradingagents.backtest import BacktestResult, BacktestSummary

_RATING_COLORS = {
    "Buy": "#2f9e44",
    "Overweight": "#66a80f",
    "Hold": "#868e96",
    "Underweight": "#e8590c",
    "Sell": "#e03131",
    "REVIEW": "#adb5bd",
}
# Ordered by how much a run should worry: a window the feed never observed is
# the one that silently costs a cell its information, while an argument the
# model invented is recovered from on the next tool call.
_REASON_COLORS = {
    "error": "#c92a2a",
    "no_data": "#e8590c",
    "no_coverage": "#e8590c",
    "vendor_unavailable": "#f08c00",
    "unsupported_request": "#adb5bd",
}


# What a decision did to the position, which is what a reader looks for on the
# line: bought in, sold out, or left it alone. REVIEW gets its own hollow mark
# because nothing was decided at all.
_ACTION_COLORS = {
    "Buy": "#2f9e44", "Overweight": "#2f9e44",
    "Hold": "#868e96",
    "Underweight": "#e03131", "Sell": "#e03131",
}


def _action_color(rating: str) -> str:
    return _ACTION_COLORS.get(rating, "#f59f00")


def _rating_color(rating: str) -> str:
    return _RATING_COLORS.get(rating, "#495057")


def _parse_pct(text: str | None) -> float | None:
    """``"+3.2%"`` -> ``0.032``; ``None``/unparseable -> ``None``."""
    if not text:
        return None
    try:
        return float(text.strip().rstrip("%")) / 100
    except ValueError:
        return None


# --- inline SVG chart primitives -------------------------------------------

def _bar_chart_svg(
    items: list[tuple[str, float, str]],
    *,
    width: int = 640,
    height: int = 240,
    value_fmt: str = "{:+.1%}",
) -> str:
    """Vertical bar chart on a zero baseline. ``items`` = (label, value, color).

    A series with no negative value sits on a baseline at the bottom and uses
    the full height; one that crosses zero gets a centred baseline instead.
    """
    if not items:
        return "<p class='empty'>Nessun dato.</p>"
    margin_l, margin_r, margin_t, margin_b = 10, 10, 20, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_abs = max(abs(v) for _, v, _ in items) or 1.0
    signed = min(v for _, v, _ in items) < 0
    zero_y = (margin_t + plot_h / 2) if signed else (margin_t + plot_h)
    span = (plot_h / 2 if signed else plot_h) - 16
    n = len(items)
    band = plot_w / n
    bar_w = min(band * 0.55, 64)

    bars = []
    for i, (label, value, color) in enumerate(items):
        cx = margin_l + band * i + band / 2
        bar_h = (abs(value) / max_abs) * span
        y = zero_y - bar_h if value >= 0 else zero_y
        label_y = (y - 6) if value >= 0 else (y + bar_h + 14)
        bars.append(
            f"<rect x='{cx - bar_w / 2:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
            f"height='{max(bar_h, 1):.1f}' fill='{color}' rx='3'>"
            f"<title>{escape(label)}: {value_fmt.format(value)}</title></rect>"
            f"<text x='{cx:.1f}' y='{label_y:.1f}' class='bar-value' "
            f"text-anchor='middle'>{value_fmt.format(value)}</text>"
            f"<text x='{cx:.1f}' y='{height - 8}' class='bar-label' "
            f"text-anchor='middle'>{escape(label)}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='chart'>"
        f"<line x1='{margin_l}' y1='{zero_y:.1f}' x2='{width - margin_r}' y2='{zero_y:.1f}' "
        f"class='axis'/>"
        f"{''.join(bars)}</svg>"
    )


def _line_chart_svg(
    series: list[tuple[str, list[float], str]],
    dates: list[str],
    *,
    markers: list[tuple[int, str]] | None = None,
    marked: list[float] | None = None,
    width: int = 760,
    height: int = 280,
) -> str:
    """Equity lines on a common scale, with a baseline at the starting value."""
    if not series or not dates:
        return "<p class='empty'>Nessuna curva da mostrare.</p>"
    margin_l, margin_r, margin_t, margin_b = 16, 16, 24, 36
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    values = [v for _, points, _ in series for v in points]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(dates)
    step = plot_w / max(n - 1, 1)

    def y_of(value: float) -> float:
        return margin_t + plot_h - (value - lo) / span * plot_h

    lines, legend = [], []
    for i, (label, points, color) in enumerate(series):
        coords = " ".join(
            f"{margin_l + step * j:.1f},{y_of(v):.1f}" for j, v in enumerate(points)
        )
        lines.append(
            f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='2'/>"
        )
        legend.append(
            f"<rect x='{margin_l + i * 150}' y='4' width='10' height='10' fill='{color}' rx='2'/>"
            f"<text x='{margin_l + i * 150 + 16}' y='13' class='bar-label'>"
            f"{escape(label)} {points[-1] - 1:+.1%}</text>"
        )

    dots = []
    for index, rating in (markers or []):
        if not 0 <= index < n:
            continue
        value = (marked or series[0][1])[index]
        hollow = rating not in _ACTION_COLORS
        dots.append(
            f"<circle cx='{margin_l + step * index:.1f}' cy='{y_of(value):.1f}' r='5' "
            f"fill='{'none' if hollow else _action_color(rating)}' "
            f"stroke='{_action_color(rating)}' stroke-width='2'>"
            f"<title>{escape(dates[index])}: {escape(rating)}</title></circle>"
        )

    ticks = []
    for j in (0, n - 1):
        ticks.append(
            f"<text x='{margin_l + step * j:.1f}' y='{height - 8}' class='bar-label' "
            f"text-anchor='{'start' if j == 0 else 'end'}'>{escape(dates[j])}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='chart'>"
        f"<line x1='{margin_l}' y1='{y_of(1.0):.1f}' x2='{width - margin_r}' "
        f"y2='{y_of(1.0):.1f}' class='axis'/>"
        f"{''.join(lines)}{''.join(dots)}{''.join(legend)}{''.join(ticks)}</svg>"
    )


def _scatter_svg(
    points: list[tuple[str, float, str, str]],
    *,
    width: int = 760,
    height: int = 260,
) -> str:
    """Timeline scatter. ``points`` = (date_label, alpha, color, tooltip), in date order."""
    if not points:
        return "<p class='empty'>Nessuna cella risolta da mostrare.</p>"
    margin_l, margin_r, margin_t, margin_b = 16, 16, 20, 36
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_abs = max(abs(v) for _, v, _, _ in points) or 1.0
    zero_y = margin_t + plot_h / 2
    n = len(points)
    step = plot_w / max(n - 1, 1) if n > 1 else 0
    label_every = max(1, n // 8)

    dots, labels = [], []
    for i, (date_label, value, color, tooltip) in enumerate(points):
        cx = margin_l + (step * i if n > 1 else plot_w / 2)
        cy = zero_y - (value / max_abs) * (plot_h / 2 - 6)
        dots.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='{color}'>"
            f"<title>{escape(tooltip)}</title></circle>"
        )
        if i % label_every == 0 or i == n - 1:
            labels.append(
                f"<text x='{cx:.1f}' y='{height - 8}' class='bar-label' "
                f"text-anchor='middle'>{escape(date_label)}</text>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='chart'>"
        f"<line x1='{margin_l}' y1='{zero_y:.1f}' x2='{width - margin_r}' y2='{zero_y:.1f}' "
        f"class='axis'/>"
        f"{''.join(dots)}{''.join(labels)}</svg>"
    )


# --- section builders --------------------------------------------------

def _metric_label(summary: BacktestSummary) -> str:
    return "Alpha medio" if summary.metric == "alpha" else "Rendimento medio"


def _rating_bars(summary: BacktestSummary) -> str:
    items = [
        (rating, score.mean_value, _rating_color(rating))
        for rating, score in summary.by_rating.items()
    ]
    return _bar_chart_svg(items)


def _hit_rate_bars(summary: BacktestSummary) -> str:
    items = [
        (rating, score.hit_rate, _rating_color(rating))
        for rating, score in summary.by_rating.items()
        if score.hit_rate is not None
    ]
    if not items:
        return "<p class='empty'>Nessun rating direzionale (solo Hold/REVIEW, o nessuna cella risolta).</p>"
    return _bar_chart_svg(items, value_fmt="{:.0%}")


def _rating_distribution(entries: list[dict]) -> str:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["rating"]] = counts.get(e["rating"], 0) + 1
    items = [(rating, float(count), _rating_color(rating)) for rating, count in counts.items()]
    return _bar_chart_svg(items, value_fmt="{:.0f}")


def _timeline(entries: list[dict]) -> str:
    # Same definition of "resolved" as summarize(): a REVIEW decision has no
    # readable rating, so its outcome scores nothing and plotting it would put
    # a point on a chart the cards above report as empty.
    resolved = [e for e in entries if not e["pending"] and e["rating"] != RATING_REVIEW]
    resolved.sort(key=lambda e: e["date"])
    points = []
    for e in resolved:
        alpha = _parse_pct(e.get("alpha"))
        if alpha is None:
            continue
        tooltip = f"{e['ticker']} {e['date']} · {e['rating']} · alpha {e['alpha']}"
        points.append((e["date"], alpha, _rating_color(e["rating"]), tooltip))
    return _scatter_svg(points)


def _strategy_section(curves: list) -> str:
    """Following the decisions, against holding the asset — plus the rule used.

    The numbers only mean anything with the assumptions beside them, so they are
    printed here rather than left in a docstring nobody opens.
    """
    if not curves:
        return "<p class='empty'>Nessuna curva: servono decisioni e prezzi per almeno un ticker.</p>"
    blocks = []
    for curve in curves:
        chart = _line_chart_svg(
            [("Strategia", curve.strategy, "#4c6ef5"), ("Buy & hold", curve.buy_hold, "#868e96")],
            curve.dates, markers=curve.markers, marked=curve.strategy,
        )
        span = (
            f"{len(curve.dates)} giorni di prezzo, {curve.dates[0]} &rarr; {curve.dates[-1]}"
            f" &middot; {len(curve.markers)} decisioni"
        )
        verdict = "#2f9e44" if curve.excess >= 0 else "#e03131"
        blocks.append(
            f"<h3>{escape(curve.ticker)}</h3>"
            f"<p class='note'>{span} &middot; "
            "<span class='dot' style='background:#2f9e44'></span> compra &middot; "
            "<span class='dot' style='background:#e03131'></span> vendi &middot; "
            "<span class='dot' style='background:#868e96'></span> mantiene &middot; "
            "<span class='dot' style='border:2px solid #f59f00'></span> non leggibile</p>"
            f"{chart}"
            "<div class='cards'>"
            f"<div class='card'><span class='n'>{curve.strategy_return:+.1%}</span>"
            "<span class='l'>strategia</span></div>"
            f"<div class='card'><span class='n'>{curve.buy_hold_return:+.1%}</span>"
            "<span class='l'>buy &amp; hold</span></div>"
            f"<div class='card'><span class='n' style='color:{verdict}'>{curve.excess:+.1%}</span>"
            "<span class='l'>differenza</span></div>"
            f"<div class='card'><span class='n'>{curve.time_in_market:.0%}</span>"
            "<span class='l'>tempo a mercato</span></div>"
            f"<div class='card'><span class='n'>{curve.changes}</span>"
            "<span class='l'>cambi di posizione</span></div>"
            "</div>"
        )
    rule = curves[0].rule
    weights = ", ".join(f"{k} {v:.0%}" for k, v in rule.items())
    costs = (f"{curves[0].cost_bps:.0f} bps per cambio di posizione"
             if curves[0].cost_bps else "nessun costo di transazione")
    carried = sum(c.carried for c in curves)
    note = (f" {carried} decisione/i senza rating leggibile hanno mantenuto la posizione."
            if carried else "")
    blocks.append(
        f"<p class='note'>Regola applicata: {escape(weights)}. Ogni decisione agisce "
        f"dalla chiusura del suo giorno; {costs}.{note} Non modella size, leva, "
        f"slippage né un portafoglio con più strumenti.</p>"
    )
    return "".join(blocks)


def _distinct_issues(fetch_issues: list[dict]) -> list[tuple[tuple, int]]:
    """One entry per distinct problem, with how many times it recurred.

    An analyst asks for one macro series per indicator, so a single missing API
    key is reported once per call; counting those as separate problems would
    make the worst-looking bar the most trivial one.
    """
    counts: dict[tuple, int] = {}
    for issue in fetch_issues:
        key = (issue["ticker"], issue["date"], issue["method"], issue["reason"], issue["detail"])
        counts[key] = counts.get(key, 0) + 1
    return list(counts.items())


def _fetch_issue_bars(fetch_issues: list[dict]) -> str:
    """Per tool, in how many cells it had a problem — not how often it retried."""
    if not fetch_issues:
        return "<p class='empty ok'>Nessun problema di fetch registrato.</p>"
    per_method: dict[str, int] = {}
    reasons: dict[str, dict[str, int]] = {}
    for (_, _, method, reason, _), _count in _distinct_issues(fetch_issues):
        per_method[method] = per_method.get(method, 0) + 1
        per_reason = reasons.setdefault(method, {})
        per_reason[reason] = per_reason.get(reason, 0) + 1
    items = []
    for method, n in sorted(per_method.items()):
        dominant = max(reasons[method].items(), key=lambda kv: kv[1])[0]
        items.append((method, float(n), _REASON_COLORS.get(dominant, "#495057")))
    return _bar_chart_svg(items, value_fmt="{:.0f}")


def _fetch_issue_table(fetch_issues: list[dict]) -> str:
    if not fetch_issues:
        return ""
    rows = []
    for (ticker, date, method, reason, detail), count in _distinct_issues(fetch_issues):
        cell = f"{ticker} · {date}" if ticker else "—"
        color = _REASON_COLORS.get(reason, "#495057")
        recurred = f" <span class='times'>×{count}</span>" if count > 1 else ""
        rows.append(
            "<tr>"
            f"<td>{escape(cell)}</td>"
            f"<td><code>{escape(method)}</code>{recurred}</td>"
            f"<td><span class='pill' style='background:{color}'>{escape(reason)}</span></td>"
            f"<td>{escape(detail)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Cella</th><th>Tool</th><th>Motivo</th><th>Dettaglio</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _failures_table(title: str, rows: list, columns: list[str]) -> str:
    if not rows:
        return ""
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{escape(c)}</th>" for c in columns)
    return (
        f"<h3>{escape(title)}</h3>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
  padding: 24px 32px 64px; background: #f8f9fa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) {
  body { background: #17191c; color: #e8e8e8; }
  .card, table, .chart { background: #202225 !important; }
  .axis { stroke: #555 !important; }
  .bar-label, .bar-value { fill: #ccc !important; }
  th { background: #2a2d31 !important; }
  td, th { border-color: #3a3d41 !important; }
}
h1 { margin-bottom: 4px; }
.subtitle { color: #868e96; margin-top: 0; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0 32px; }
.card { background: #fff; border-radius: 10px; padding: 16px 20px; min-width: 140px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.card .n { font-size: 28px; font-weight: 700; display: block; }
.card .l { color: #868e96; font-size: 13px; }
section { margin-bottom: 40px; }
h2 { font-size: 18px; border-bottom: 1px solid #dee2e6; padding-bottom: 8px; }
.chart { background: #fff; border-radius: 10px; padding: 8px; }
.axis { stroke: #ccc; stroke-width: 1; }
.bar-label { font-size: 11px; fill: #666; }
.bar-value { font-size: 11px; fill: #333; font-weight: 600; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; background: #fff;
  border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e9ecef; font-size: 13px; }
th { background: #f1f3f5; }
code { font-size: 12px; }
.pill { color: #fff; border-radius: 999px; padding: 2px 8px; font-size: 12px; }
.empty { color: #868e96; font-style: italic; }
.empty.ok { color: #2f9e44; }
.times { color: #868e96; font-size: 12px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  vertical-align: middle; }
.note { color: #868e96; font-size: 13px; }
"""


def _holding_note(entries: list[dict]) -> str:
    """The window the outcomes were measured over, in this report's own words.

    ``BacktestSummary.holding`` renders an English phrase for the console; the
    entries carry the raw ``5d``, so the page reads in one language.
    """
    windows = sorted({
        e["holding"][:-1] for e in entries
        if not e["pending"] and (e.get("holding") or "").endswith("d")
    })
    if not windows:
        return "la finestra configurata"
    return ", ".join(f"{w} giorni di trading" for w in windows)


def write_html_report(
    result: BacktestResult,
    summary: BacktestSummary,
    entries: list[dict],
    config: dict,
) -> Path:
    """Write a self-contained HTML report next to the run's decision log.

    Returns the report path (``result.log_path.parent / "report.html"``).
    """
    tickers = sorted({e["ticker"] for e in entries}) or ["—"]
    dates = sorted({e["date"] for e in entries})
    date_range = f"{dates[0]} → {dates[-1]}" if dates else "—"
    model = f"{config.get('llm_provider', '?')} / {config.get('deep_think_llm', '?')}, {config.get('quick_think_llm', '?')}"

    body = f"""
<h1>Backtest report</h1>
<p class="subtitle">Run <code>{escape(result.run_id)}</code> · {escape(', '.join(tickers))}
  · {escape(date_range)} · {escape(model)}
  · generato {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="cards">
  <div class="card"><span class="n">{result.cells_run}</span><span class="l">celle eseguite</span></div>
  <div class="card"><span class="n">{result.skipped}</span><span class="l">celle saltate</span></div>
  <div class="card"><span class="n">{summary.resolved}</span><span class="l">risolte</span></div>
  <div class="card"><span class="n">{summary.pending}</span><span class="l">pending</span></div>
  <div class="card"><span class="n">{summary.unscored}</span><span class="l">unscored (REVIEW)</span></div>
</div>

<section>
  <h2>Seguendo le decisioni, contro comprare e tenere</h2>
  {_strategy_section(result.curves)}
</section>

<section>
  <h2>{_metric_label(summary)} per rating</h2>
  {_rating_bars(summary)}
</section>

<section>
  <h2>Hit rate direzionale per rating</h2>
  {_hit_rate_bars(summary)}
</section>

<section>
  <h2>Alpha nel tempo per cella risolta</h2>
  {_timeline(entries)}
</section>

<section>
  <h2>Distribuzione dei rating emessi</h2>
  {_rating_distribution(entries)}
</section>

<section>
  <h2>Qualità dei dati fetchati</h2>
  {_fetch_issue_bars(result.fetch_issues)}
  {_fetch_issue_table(result.fetch_issues)}
</section>

<section>
  <h2>Celle e settlement falliti</h2>
  {_failures_table("Celle fallite", result.failures, ["Ticker", "Data", "Errore"])}
  {_failures_table("Settlement falliti", result.settlement_failures, ["Ticker", "Errore"])}
  {"<p class='empty ok'>Nessun fallimento.</p>" if not result.failures and not result.settlement_failures else ""}
</section>

<p class="note">{_metric_label(summary)} misurato su {escape(_holding_note(entries))} dopo ogni
  data di analisi{" contro il benchmark dello strumento" if summary.metric == "alpha" else ""}. Un solo campionamento LLM per cella: numeri indicativi, non riproducibili al bit.</p>
"""

    html = f"<!doctype html><html><head><meta charset='utf-8'>" \
           f"<title>Backtest {escape(result.run_id)}</title><style>{_CSS}</style></head>" \
           f"<body>{body}</body></html>"

    report_path = result.log_path.parent / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
