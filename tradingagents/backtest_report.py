"""Self-contained HTML report for a backtest run.

Single file, no external assets: inline CSS and hand-rolled inline SVG charts
(stdlib only — the project ships no plotting/HTML dependency and this keeps it
that way), so the report opens and reads correctly with no network access.

Colour carries one job per chart. The two equity lines are an identity pair
(validated categorical slots); a rating is coloured by its *direction* only, with
the tier named in the label beside it, because green and red sit in the
colourblind warn band — so every mark that carries direction also carries a
shape, and no reading depends on hue alone.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from tradingagents.agents.utils.rating import RATING_REVIEW

if TYPE_CHECKING:
    from tradingagents.backtest import BacktestResult, BacktestSummary

# A rating's direction, not its tier: the tier is always written beside the mark.
# Green/red are ΔE 7.2 apart under protanopia, which is legal only with a second
# channel, so every mark using these also differs in shape.
_BULLISH, _BEARISH = ("Buy", "Overweight"), ("Underweight", "Sell")


def _direction_role(rating: str) -> str:
    if rating in _BULLISH:
        return "up"
    if rating in _BEARISH:
        return "down"
    return "unknown" if rating == RATING_REVIEW else "flat"


def _rating_color(rating: str) -> str:
    return f"var(--dir-{_direction_role(rating)})"


# Reasons are states, not series: they wear the reserved status ink and always
# ship with their name in the row beside them.
_REASON_ROLES = {
    "error": "critical",
    "no_data": "serious",
    "no_coverage": "warning",
    "vendor_unavailable": "serious",
    "unsupported_request": "muted",
}


def _reason_color(reason: str) -> str:
    return f"var(--status-{_REASON_ROLES.get(reason, 'muted')})"


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
    width: int = 720,
    height: int = 260,
    value_fmt: str = "{:+.1%}",
) -> str:
    """Vertical bar chart on a zero baseline. ``items`` = (label, value, color).

    A series with no negative value sits on a baseline at the bottom and uses
    the full height; one that crosses zero gets a centred baseline instead.
    """
    if not items:
        return "<p class='empty'>Nessun dato.</p>"
    margin_l, margin_r, margin_t, margin_b = 12, 12, 26, 44
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_abs = max(abs(v) for _, v, _ in items) or 1.0
    signed = min(v for _, v, _ in items) < 0
    zero_y = (margin_t + plot_h / 2) if signed else (margin_t + plot_h)
    span = (plot_h / 2 if signed else plot_h) - 18
    n = len(items)
    band = plot_w / n
    bar_w = min(band * 0.5, 56)

    bars = []
    for i, (label, value, color) in enumerate(items):
        cx = margin_l + band * i + band / 2
        bar_h = (abs(value) / max_abs) * span
        y = zero_y - bar_h if value >= 0 else zero_y
        label_y = (y - 8) if value >= 0 else (y + bar_h + 16)
        bars.append(
            f"<rect x='{cx - bar_w / 2:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
            f"height='{max(bar_h, 2):.1f}' fill='{color}' rx='4'>"
            f"<title>{escape(label)}: {value_fmt.format(value)}</title></rect>"
            f"<text x='{cx:.1f}' y='{label_y:.1f}' class='mark-value' "
            f"text-anchor='middle'>{value_fmt.format(value)}</text>"
            f"<text x='{cx:.1f}' y='{height - 10}' class='mark-label' "
            f"text-anchor='middle'>{escape(label)}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='chart' "
        f"role='img'><line x1='{margin_l}' y1='{zero_y:.1f}' x2='{width - margin_r}' "
        f"y2='{zero_y:.1f}' class='axis'/>{''.join(bars)}</svg>"
    )


def _marker_shape(rating: str, cx: float, cy: float, color: str) -> str:
    """Shape is the channel that survives colourblindness; colour reinforces it."""
    role = _direction_role(rating)
    if role == "up":
        pts = f"{cx:.1f},{cy - 6:.1f} {cx + 5.5:.1f},{cy + 4:.1f} {cx - 5.5:.1f},{cy + 4:.1f}"
        return f"<polygon points='{pts}' fill='{color}' class='marker'/>"
    if role == "down":
        pts = f"{cx:.1f},{cy + 6:.1f} {cx + 5.5:.1f},{cy - 4:.1f} {cx - 5.5:.1f},{cy - 4:.1f}"
        return f"<polygon points='{pts}' fill='{color}' class='marker'/>"
    if role == "unknown":
        return (f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='none' "
                f"stroke='{color}' stroke-width='2' class='marker'/>")
    return f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='{color}' class='marker'/>"


def _line_chart_svg(
    series: list[tuple[str, list[float], str]],
    dates: list[str],
    *,
    markers: list[tuple[int, str]] | None = None,
    marked: list[float] | None = None,
    width: int = 860,
    height: int = 320,
) -> str:
    """Equity lines on a common scale, with a baseline at the starting value."""
    if not series or not dates:
        return "<p class='empty'>Nessuna curva da mostrare.</p>"
    margin_l, margin_r, margin_t, margin_b = 16, 46, 18, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    values = [v for _, points, _ in series for v in points]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.12 or 0.02
    lo, hi = lo - pad, hi + pad
    span = (hi - lo) or 1.0
    n = len(dates)
    step = plot_w / max(n - 1, 1)

    def y_of(value: float) -> float:
        return margin_t + plot_h - (value - lo) / span * plot_h

    # Gridlines at round moves from the starting capital, labelled once on the right.
    grid = []
    tick = 0.05 if span < 0.5 else 0.25
    level = round(lo / tick) * tick
    while level <= hi:
        if lo < level < hi:
            y = y_of(level)
            grid.append(
                f"<line x1='{margin_l}' y1='{y:.1f}' x2='{width - margin_r}' "
                f"y2='{y:.1f}' class='grid'/>"
                f"<text x='{width - margin_r + 6}' y='{y + 4:.1f}' class='mark-label'>"
                f"{level - 1:+.0%}</text>"
            )
        level += tick

    lines = []
    for label, points, color in series:
        coords = " ".join(
            f"{margin_l + step * j:.1f},{y_of(v):.1f}" for j, v in enumerate(points)
        )
        lines.append(
            f"<polyline points='{coords}' fill='none' stroke='{color}' "
            f"stroke-width='2' stroke-linejoin='round' stroke-linecap='round'>"
            f"<title>{escape(label)}</title></polyline>"
        )

    dots = []
    for index, rating in (markers or []):
        if not 0 <= index < n:
            continue
        value = (marked or series[0][1])[index]
        dots.append(
            "<g>"
            + _marker_shape(rating, margin_l + step * index, y_of(value), _rating_color(rating))
            + f"<title>{escape(dates[index])}: {escape(rating)}</title></g>"
        )

    ticks = "".join(
        f"<text x='{margin_l + step * j:.1f}' y='{height - 12}' class='mark-label' "
        f"text-anchor='{'start' if j == 0 else 'end'}'>{escape(dates[j])}</text>"
        for j in (0, n - 1)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='chart' role='img'>"
        f"{''.join(grid)}"
        f"<line x1='{margin_l}' y1='{y_of(1.0):.1f}' x2='{width - margin_r}' "
        f"y2='{y_of(1.0):.1f}' class='axis'/>"
        f"{''.join(lines)}{''.join(dots)}{ticks}</svg>"
    )


def _scatter_svg(
    points: list[tuple[str, float, str, str]],
    *,
    width: int = 720,
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
        cy = zero_y - (value / max_abs) * (plot_h / 2 - 10)
        dots.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5' fill='{color}' class='marker'>"
            f"<title>{escape(tooltip)}</title></circle>"
        )
        if i % label_every == 0 or i == n - 1:
            labels.append(
                f"<text x='{cx:.1f}' y='{height - 10}' class='mark-label' "
                f"text-anchor='middle'>{escape(date_label)}</text>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' class='chart' role='img'>"
        f"<line x1='{margin_l}' y1='{zero_y:.1f}' x2='{width - margin_r}' "
        f"y2='{zero_y:.1f}' class='axis'/>{''.join(dots)}{''.join(labels)}</svg>"
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


def _tile(value: str, label: str) -> str:
    return (f"<div class='tile'><span class='tile-n'>{escape(value)}</span>"
            f"<span class='tile-l'>{escape(label)}</span></div>")


_MARKER_KEY = (
    "<span class='key'><svg viewBox='0 0 12 12' class='key-mark'>"
    "<polygon points='6,1 11,10 1,10' fill='var(--dir-up)'/></svg>compra</span>"
    "<span class='key'><svg viewBox='0 0 12 12' class='key-mark'>"
    "<polygon points='6,11 11,2 1,2' fill='var(--dir-down)'/></svg>vendi</span>"
    "<span class='key'><svg viewBox='0 0 12 12' class='key-mark'>"
    "<circle cx='6' cy='6' r='4.5' fill='var(--dir-flat)'/></svg>mantiene</span>"
    "<span class='key'><svg viewBox='0 0 12 12' class='key-mark'>"
    "<circle cx='6' cy='6' r='4' fill='none' stroke='var(--dir-unknown)' "
    "stroke-width='2'/></svg>non leggibile</span>"
)


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
            [("Strategia", curve.strategy, "var(--series-1)"),
             ("Buy & hold", curve.buy_hold, "var(--series-2)")],
            curve.dates, markers=curve.markers, marked=curve.strategy,
        )
        tone = "var(--dir-up)" if curve.excess >= 0 else "var(--dir-down)"
        verdict = "ha battuto" if curve.excess >= 0 else "ha fatto peggio di"
        blocks.append(
            "<div class='panel'>"
            f"<div class='panel-head'><h3>{escape(curve.ticker)}</h3>"
            f"<p class='sub'>{len(curve.dates)} giorni di prezzo &middot; "
            f"{escape(curve.dates[0])} &rarr; {escape(curve.dates[-1])} &middot; "
            f"{len(curve.markers)} decisioni</p></div>"
            f"<p class='hero' style='color:{tone}'>{curve.excess:+.1%}</p>"
            f"<p class='hero-sub'>seguire le decisioni <strong>{verdict}</strong> "
            "comprare e tenere</p>"
            "<div class='legend'>"
            "<span class='key'><i class='swatch' style='background:var(--series-1)'></i>"
            f"Strategia {curve.strategy_return:+.1%}</span>"
            "<span class='key'><i class='swatch' style='background:var(--series-2)'></i>"
            f"Buy &amp; hold {curve.buy_hold_return:+.1%}</span>"
            f"</div>{chart}"
            f"<div class='legend legend-marks'>{_MARKER_KEY}</div>"
            "<div class='tiles'>"
            + _tile(f"{curve.time_in_market:.0%}", "tempo a mercato")
            + (_tile(f"{curve.average_exposure:.0%}", "esposizione media")
               if curve.trades is not None else "")
            + _tile(str(curve.changes), "operazioni" if curve.trades is not None
                    else "cambi di posizione")
            + _tile(f"{curve.max_drawdown('strategy'):.1%}", "drawdown strategia")
            + _tile(f"{curve.max_drawdown('buy_hold'):.1%}", "drawdown buy & hold")
            + "</div></div>"
        )
    costs = (f"{curves[0].cost_bps:.0f} bps per cambio di posizione"
             if curves[0].cost_bps else "nessun costo di transazione")
    carried = sum(c.carried for c in curves)
    note = (f" {carried} decisione/i senza rating leggibile hanno mantenuto la posizione."
            if carried else "")
    if curves[0].trades is not None:
        blocks.append(
            f"<p class='note'>Regola applicata, le decisioni come ordini: "
            f"{escape(_describe_trades(curves[0].trades))}. Si parte tutto in liquidità; "
            "gli acquisti sono limitati alla liquidità e le vendite a quanto posseduto, "
            "quindi niente leva né vendite allo scoperto (posizione tra 0% e 100%). Tra "
            "un ordine e l'altro la posizione non viene ribilanciata: il suo peso segue "
            f"il prezzo. Ogni decisione agisce dalla chiusura del suo giorno; {costs}.{note} "
            "Non modella slippage né un portafoglio con più strumenti.</p>"
        )
        return "".join(blocks)
    weights = ", ".join(f"{k} {v:.0%}" for k, v in curves[0].rule.items())
    blocks.append(
        f"<p class='note'>Regola applicata: {escape(weights)}. Ogni decisione agisce "
        f"dalla chiusura del suo giorno; {costs}.{note} Non modella size, leva, "
        f"slippage né un portafoglio con più strumenti.</p>"
    )
    return "".join(blocks)


def _describe_trades(trades: dict[str, float]) -> str:
    """``Buy compra il 100% del patrimonio, ..., Hold non muove nulla``."""
    parts = [
        f"{rating} {'compra' if fraction > 0 else 'vende'} il {abs(fraction):.0%} del patrimonio"
        for rating, fraction in trades.items() if fraction
    ]
    idle = [r for r in ("Buy", "Overweight", "Hold", "Underweight", "Sell") if not trades.get(r)]
    if idle:
        parts.append(f"{', '.join(idle)} non {'muove' if len(idle) == 1 else 'muovono'} nulla")
    return ", ".join(parts)


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
        items.append((method, float(n), _reason_color(dominant)))
    return _bar_chart_svg(items, value_fmt="{:.0f}")


def _fetch_issue_table(fetch_issues: list[dict]) -> str:
    if not fetch_issues:
        return ""
    rows = []
    for (ticker, date, method, reason, detail), count in _distinct_issues(fetch_issues):
        cell = f"{ticker} · {date}" if ticker else "—"
        recurred = f" <span class='times'>×{count}</span>" if count > 1 else ""
        rows.append(
            "<tr>"
            f"<td>{escape(cell)}</td>"
            f"<td><code>{escape(method)}</code>{recurred}</td>"
            f"<td><span class='pill' style='background:{_reason_color(reason)}'>"
            f"{escape(reason)}</span></td>"
            f"<td class='detail'>{escape(detail)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Cella</th><th>Tool</th><th>Motivo</th><th>Dettaglio</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _decisions_table(entries: list[dict]) -> str:
    """Every decision the sweep took, beside the outcome that settled it."""
    if not entries:
        return "<p class='empty'>Nessuna decisione registrata.</p>"
    rows = []
    for e in sorted(entries, key=lambda x: (x["ticker"], x["date"])):
        if e["rating"] == RATING_REVIEW:
            state = ("<span class='pill' style='background:var(--status-muted)'>"
                     "non valutabile</span>")
        elif e["pending"]:
            state = "<span class='pill pill-quiet'>in attesa</span>"
        else:
            state = f"risolta {escape(e.get('resolved') or '')}"
        rows.append(
            "<tr>"
            f"<td>{escape(e['date'])}</td>"
            f"<td>{escape(e['ticker'])}</td>"
            f"<td><span class='tag' style='color:{_rating_color(e['rating'])}'>"
            f"{escape(e['rating'])}</span></td>"
            f"<td class='num'>{escape(e.get('raw') or '—')}</td>"
            f"<td class='num'>{escape(e.get('alpha') or '—')}</td>"
            f"<td>{state}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Data</th><th>Ticker</th><th>Rating</th>"
        "<th class='num'>Rendimento</th><th class='num'>Alpha</th><th>Stato</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
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


_MISMATCH_LABELS = {
    "unrecorded": "celle eseguite prima che esistesse un manifesto",
    "code": "codice",
    "model": "modello o parametri di generazione",
    "server": "server Ollama (pesi, contesto o KV cache)",
    "pipeline": "pipeline (analisti, round, orizzonte, vendor)",
    "sweep": "impostazioni dello sweep (cache, finestre, lezioni)",
}


def _mismatch_banner(mismatch: list[str]) -> str:
    if not mismatch:
        return ""
    what = "; ".join(_MISMATCH_LABELS.get(k, k) for k in mismatch)
    return (
        "<div class='panel warn-panel'><h3>Celle prodotte in condizioni diverse</h3>"
        f"<p>Questo sweep &egrave; stato ripreso con: {escape(what)}. Le celle non sono tutte "
        "confrontabili tra loro: per confrontare configurazioni o modelli usa un nuovo "
        "<code>--run-id</code>.</p></div>"
    )


def _yes_no(value) -> str:
    return "s&igrave;" if value else "no"


def _manifest_section(manifest: dict) -> str:
    """The conditions a sweep ran under, so two reports can be checked for comparability."""
    if not manifest:
        return "<p class='empty'>Nessun manifesto registrato per questo run.</p>"
    code = manifest.get("code") or {}
    model = manifest.get("model") or {}
    server = manifest.get("server") or {}
    grid = manifest.get("grid") or {}
    pipeline = manifest.get("pipeline") or {}
    sweep = manifest.get("sweep") or {}
    strategy = manifest.get("strategy") or {}

    commit = manifest.get("commit")
    code_cell = f"<code>{escape(commit[:10])}</code>" if commit else "sconosciuto (non &egrave; un checkout git)"
    if code.get("uncommitted"):
        code_cell += (" <span class='warn'>+ modifiche non committate "
                      f"(impronta <code>{escape(code['uncommitted'])}</code>)</span>")

    deep, quick = model.get("deep_think_llm"), model.get("quick_think_llm")
    names = (f"{escape(str(deep))} (deep e quick)" if deep == quick
             else f"{escape(str(deep))} (deep), {escape(str(quick))} (quick)")
    temperature = model.get("temperature")
    max_tokens = model.get("max_tokens")
    rows = [
        ("Codice", code_cell),
        ("Modello", f"{escape(str(model.get('provider')))} &middot; {names}"),
        ("Generazione",
         f"temperatura {escape(str(temperature)) if temperature is not None else 'default del provider'}"
         f" &middot; max token in uscita {escape(str(max_tokens)) if max_tokens else 'nessun limite'}"),
    ]
    for name, info in (server.get("models") or {}).items():
        rows.append((f"Pesi {escape(name)}", " &middot; ".join(escape(str(v)) for v in (
            f"digest {info.get('digest')}", info.get("parameters"), info.get("quantization"),
            f"contesto {info.get('num_ctx')} token",
        ))))
    if server:
        source = "dal log del server" if server.get("read_from") == "server log" else "dall'ambiente del client"
        rows.append(("Server Ollama",
                     f"KV cache {escape(str(server.get('kv_cache_type')))} &middot; flash attention "
                     f"{escape(str(server.get('flash_attention')))} <span class='times'>({source})</span>"))
    every = grid.get("every_days")
    step = ("ogni giorno" if every == 1 else f"ogni {every} giorni" if every else "passo irregolare")
    holding = pipeline.get("holding_period_days")
    rows += [
        ("Griglia",
         f"{escape(str(grid.get('first')))} &rarr; {escape(str(grid.get('last')))} &middot; "
         f"{grid.get('cells')} celle &middot; {step} &middot; {escape(str(grid.get('asset_type')))}"),
        ("Pipeline",
         f"analisti {escape(', '.join(pipeline.get('analysts') or []))} &middot; "
         f"debate {pipeline.get('max_debate_rounds')} round &middot; rischio "
         f"{pipeline.get('max_risk_discuss_rounds')} round &middot; orizzonte "
         f"{holding} {'giorno' if holding == 1 else 'giorni'} di trading"),
        ("Vendor dati", escape(", ".join(
            f"{category}={vendor or 'off'}" for category, vendor in (pipeline.get("data_vendors") or {}).items()
        ) + "".join(f", {tool}={vendor}" for tool, vendor in (pipeline.get("tool_vendors") or {}).items()))),
        ("Sweep",
         f"cache dei fetch {_yes_no(sweep.get('cache_tool_fetches'))} &middot; finestre fisse "
         f"{_yes_no(sweep.get('canonical_tool_windows'))} (news {sweep.get('news_lookback_days')} g, "
         f"prezzi {sweep.get('price_lookback_days')} g) &middot; lezioni dalle decisioni passate "
         f"{_yes_no(sweep.get('learn_from_past_decisions'))}"),
        ("Strategia", escape(
            "ordini: " + _describe_trades(strategy["trades"]) if strategy.get("trades")
            else ", ".join(f"{rating} {weight:g}"
                           for rating, weight in (strategy.get("positions") or {}).items())
        ) + f" &middot; costo {strategy.get('cost_bps') or 0:g} bps"),
    ]
    body = "".join(f"<tr><th scope='row'>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return (
        f"<table class='kv'><tbody>{body}</tbody></table>"
        "<p class='note'>Due sweep sono confrontabili quando queste righe coincidono tranne il "
        "modello. Il manifesto completo &egrave; in <code>manifest.json</code>, accanto al log.</p>"
    )


_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,.10);
  --series-1: #2a78d6; --series-2: #eb6834;
  --dir-up: #008300; --dir-down: #e34948; --dir-flat: #898781; --dir-unknown: #eda100;
  --status-critical: #d03b3b; --status-serious: #ec835a;
  --status-warning: #fab219; --status-muted: #898781;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
    --series-1: #3987e5; --series-2: #d95926;
    --dir-down: #e66767;
  }
}
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 40px 24px 80px; background: var(--plane); color: var(--ink);
  font-size: 14px; line-height: 1.55; }
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 26px; letter-spacing: -.02em; margin: 0 0 4px; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--ink-muted); font-weight: 600; margin: 0 0 12px; }
h3 { font-size: 15px; margin: 0; }
.sub { color: var(--ink-2); margin: 2px 0 0; font-size: 13px; }
section { margin: 36px 0; }
.panel { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 14px; padding: 20px; margin-bottom: 16px; }
.panel-head { margin-bottom: 8px; }
.hero { font-size: 46px; font-weight: 700; letter-spacing: -.03em;
  margin: 12px 0 0; line-height: 1; }
.hero-sub { color: var(--ink-2); margin: 6px 0 14px; }
.tiles { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.tile { background: var(--plane); border: 1px solid var(--ring); border-radius: 10px;
  padding: 10px 14px; flex: 1 1 120px; }
.tile-n { display: block; font-size: 20px; font-weight: 650; letter-spacing: -.01em; }
.tile-l { color: var(--ink-muted); font-size: 12px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0 4px;
  color: var(--ink-2); font-size: 13px; }
.legend-marks { margin: 2px 0 0; color: var(--ink-muted); font-size: 12px; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.key-mark { width: 12px; height: 12px; }
.chart { display: block; width: 100%; margin: 4px 0; overflow: visible; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.mark-label { font-size: 11px; fill: var(--ink-muted); }
.mark-value { font-size: 11px; fill: var(--ink-2); font-weight: 600; }
.marker { stroke: var(--surface); stroke-width: 2; paint-order: stroke; }
table { width: 100%; border-collapse: collapse; margin-top: 10px;
  background: var(--surface); border: 1px solid var(--ring);
  border-radius: 12px; overflow: hidden; font-size: 13px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--ring); }
th { color: var(--ink-muted); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: .05em; }
tbody tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.detail { color: var(--ink-2); }
.tag { font-weight: 650; }
code { font-size: 12px; color: var(--ink-2); }
.pill { color: #fff; border-radius: 999px; padding: 2px 9px; font-size: 12px;
  white-space: nowrap; }
.pill-quiet { background: var(--axis); color: var(--ink); }
.times { color: var(--ink-muted); font-size: 12px; }
.empty { color: var(--ink-muted); font-style: italic; }
.empty.ok { color: var(--dir-up); font-style: normal; }
.note { color: var(--ink-muted); font-size: 12.5px; max-width: 78ch; }
.kv th { width: 26%; text-transform: none; letter-spacing: 0; font-size: 12.5px;
  vertical-align: top; }
.warn { color: var(--status-serious); }
.warn-panel { border-left: 4px solid var(--status-warning); }
.warn-panel p { margin: 6px 0 0; color: var(--ink-2); }
@media (max-width: 560px) {
  body { padding: 24px 14px 60px; }
  .hero { font-size: 36px; }
}
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
    filename: str = "report.html",
) -> Path:
    """Write a self-contained HTML report next to the run's decision log.

    Returns the report path (``result.log_path.parent / "report.html"``).
    """
    tickers = sorted({e["ticker"] for e in entries}) or ["—"]
    dates = sorted({e["date"] for e in entries})
    date_range = f"{dates[0]} → {dates[-1]}" if dates else "—"
    model = f"{config.get('llm_provider', '?')} / {config.get('deep_think_llm', '?')}, {config.get('quick_think_llm', '?')}"
    commit = result.manifest.get("commit")
    code = f" &middot; commit <code>{escape(commit[:7])}</code>" if commit else ""
    if (result.manifest.get("code") or {}).get("uncommitted"):
        code += " <span class='warn'>+ modifiche locali</span>"

    body = f"""
<main>
<header>
  <h1>Backtest {escape(', '.join(tickers))}</h1>
  <p class="sub">Run <code>{escape(result.run_id)}</code> &middot; {escape(date_range)}
    &middot; {escape(model)}{code} &middot; generato {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</header>
{_mismatch_banner(result.manifest_mismatch)}

<section>
  <h2>Seguendo le decisioni{" come ordini" if any(c.trades is not None for c in result.curves) else ""}, contro comprare e tenere</h2>
  {_strategy_section(result.curves)}
</section>

<section>
  <h2>Copertura del run</h2>
  <div class="tiles">
    {_tile(str(result.cells_run), "celle eseguite")}
    {_tile(str(result.skipped), "celle saltate")}
    {_tile(str(summary.resolved), "risolte")}
    {_tile(str(summary.pending), "in attesa")}
    {_tile(str(summary.unscored), "non valutabili")}
  </div>
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
  <h2>Ogni decisione presa</h2>
  {_decisions_table(entries)}
</section>

<section>
  <h2>Qualit&agrave; dei dati fetchati</h2>
  {_fetch_issue_bars(result.fetch_issues)}
  {_fetch_issue_table(result.fetch_issues)}
</section>

<section>
  <h2>Celle e settlement falliti</h2>
  {_failures_table("Celle fallite", result.failures, ["Ticker", "Data", "Errore"])}
  {_failures_table("Settlement falliti", result.settlement_failures, ["Ticker", "Errore"])}
  {"<p class='empty ok'>Nessun fallimento.</p>" if not result.failures and not result.settlement_failures else ""}
</section>

<section>
  <h2>Condizioni del run</h2>
  {_manifest_section(result.manifest)}
</section>

<p class="note">{_metric_label(summary)} misurato su {escape(_holding_note(entries))} dopo ogni
  data di analisi{" contro il benchmark dello strumento" if summary.metric == "alpha" else ""}.
  Un solo campionamento LLM per cella: numeri indicativi, non riproducibili al bit.</p>
</main>
"""

    html = f"<!doctype html><html lang='it'><head><meta charset='utf-8'>" \
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>" \
           f"<title>Backtest {escape(result.run_id)}</title><style>{_CSS}</style></head>" \
           f"<body>{body}</body></html>"

    report_path = result.log_path.parent / filename
    report_path.write_text(html, encoding="utf-8")
    return report_path
