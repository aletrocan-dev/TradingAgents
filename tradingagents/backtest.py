"""Run the graph over a grid of tickers and dates, and score what came back.

One run yields one decision, so it cannot say whether the system decides well.
This runs the same machinery over many (ticker, date) cells and reads the
aggregate. The decision log is the results table: every run already records its
rating and later settles it with realized and alpha return against the
instrument's regional benchmark, so there is nothing to record separately.

Scope: this evaluates decision quality. It is not a portfolio simulator, and
must not grow one. Turning a rating into a filled order needs a quantity, a fill
price and a cash ledger, none of which the system has; inventing them here would
put an execution model behind an evaluation tool. Cells are therefore
independent, and a portfolio, when given, is the same standing book for every
cell rather than a position carried forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATING_REVIEW
from tradingagents.backtest_report import write_html_report
from tradingagents.dataflows import fetch_issues
from tradingagents.dataflows.utils import get_current_date, safe_ticker_component
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.strategy_curve import StrategyCurve, build_curves

logger = logging.getLogger(__name__)


def iter_grid(start_date: str, end_date: str, every_n_days: int = 1) -> list[str]:
    """Analysis dates from ``start_date``, never past today.

    A future date has no outcome to settle against, and the graph rejects one, so
    the grid stops at the present rather than producing cells that cannot score.
    """
    start, end = _canonical(start_date), _canonical(end_date)
    if every_n_days < 1:
        raise ValueError("every_n_days must be at least 1")
    if end < start:
        raise ValueError(f"the grid ends before it starts: {end_date} is before {start_date}")

    last = min(end, datetime.strptime(get_current_date(), "%Y-%m-%d"))
    dates, cursor = [], start
    while cursor <= last:
        dates.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=every_n_days)
    return dates


def _canonical(date: str) -> datetime:
    """Parse a grid bound, rejecting anything the run date would also reject."""
    try:
        parsed = datetime.strptime(str(date), "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid dates must be in YYYY-MM-DD format, got {date!r}") from exc
    if parsed.strftime("%Y-%m-%d") != str(date):
        raise ValueError(f"grid dates must be in YYYY-MM-DD format, got {date!r}")
    return parsed


def _outcome(entry: dict, metric: str = "alpha") -> float | None:
    """A settled entry's outcome under ``metric``, or None when it has not settled.

    The log stores both the raw and the alpha return as a percentage rounded to
    one decimal, so aggregates here are accurate to 0.1 of a percentage point,
    not to the raw quote.
    """
    text = (entry.get(metric) or "").strip().rstrip("%")
    try:
        return float(text) / 100
    except ValueError:
        return None


def _grid_holding_days(dates: list[str], asset_type: str) -> int | None:
    """Bars a decision owns before the next one supersedes it, or ``None`` when
    the grid is uneven and no single horizon describes it.

    ``_fetch_returns`` counts bars, and the grid steps calendar days: crypto
    prints a bar every day, so the two coincide, while an equity prints five a
    week and a seven-day step is five bars.
    """
    if len(dates) < 2:
        return None
    gaps = {
        (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days
        for a, b in zip(dates, dates[1:], strict=False)
    }
    if len(gaps) != 1:
        return None
    gap = gaps.pop()
    return max(1, gap if asset_type == "crypto" else round(gap * 5 / 7))


@dataclass
class BacktestResult:
    run_id: str
    log_path: Path
    cells_run: int = 0
    skipped: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    settlement_failures: list[tuple[str, str]] = field(default_factory=list)
    # Data fetches that degraded to "no data"/"vendor unavailable" or errored,
    # one dict per occurrence (see tradingagents.dataflows.fetch_issues).
    fetch_issues: list[dict] = field(default_factory=list)
    # One per ticker: following the decisions, against holding the asset.
    curves: list[StrategyCurve] = field(default_factory=list)
    # Which outcome the sweep is scored on; see TradingAgentsGraph.scoring_metric.
    metric: str = "alpha"
    # Set once write_html_report() has run at the end of run_backtest().
    report_path: Path | None = None


# What each rating claims will happen, so an outcome can be scored against it.
# Hold claims no direction, so nothing about alpha proves it right or wrong.
_DIRECTION = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}


@dataclass
class RatingScore:
    count: int
    hit_rate: float | None
    mean_value: float   # alpha or raw return, per BacktestSummary.metric


@dataclass
class BacktestSummary:
    resolved: int
    pending: int
    by_rating: dict[str, RatingScore]
    unscored: int = 0
    holding: str = ""
    # Which outcome the scores above are computed on. Two reports are only
    # comparable when this matches, so render() and the HTML both name it.
    metric: str = "alpha"

    def render(self) -> str:
        lines = [f"Resolved cells: {self.resolved} · pending: {self.pending}"
                 + (f" · unscored: {self.unscored}" if self.unscored else "")]
        label = "alpha" if self.metric == "alpha" else "return"
        against = " vs the benchmark" if self.metric == "alpha" else ""
        for rating, score in self.by_rating.items():
            called = (f"called the direction {score.hit_rate:.0%}"
                      if score.hit_rate is not None else "no direction claimed")
            lines.append(
                f"- {rating}: n={score.count}, {called}, "
                f"mean {label} {score.mean_value:+.2%}{against}"
            )
        lines.append("")
        if self.pending:
            lines.append("Pending cells are not scored above; re-run to settle them.")
        lines.append(
            f"{label.capitalize()} is measured over {self.holding} after each analysis date. "
            "One model sampling per cell, and text feeds are not archived, so "
            "these figures are indicative rather than repeatable."
        )
        return "\n".join(lines)


def run_backtest(
    tickers: list[str],
    dates: list[str],
    config: dict,
    asset_type: str = "stock",
    portfolio=None,
    selected_analysts=("market", "social", "news", "fundamentals"),
    run_id: str | None = None,
    cache_fetches: bool = True,
    canonical_windows: bool = True,
) -> BacktestResult:
    """Analyze every ticker on every date, into a decision log of this run's own.

    The live log stays untouched: a sweep would otherwise flood the context that
    real runs read back. Cells already in this run's log are skipped, so an
    interrupted sweep resumes by being run again.

    ``cache_fetches`` (default on) caches every data-vendor tool call under the
    shared ``data_cache_dir`` (not per-run_id), keyed on the exact call and
    never on the model/provider, so a later sweep over the same cells — with a
    different model, even run weeks apart — reads identical inputs instead of
    re-hitting live vendors (news/social included). Fetch failures/unavailable
    data are collected into the result regardless of this flag.

    ``canonical_windows`` (default on) pins every tool's window length to the
    configured one, so two models are compared on the same question rather than
    on which one asked to look further back. It is deliberately a sweep-only
    setting: a single live run is the product and chooses its own windows.

    The horizon each decision is judged over is taken from the grid rather than
    from ``holding_period_days``: the next cell supersedes the previous decision,
    so the gap between them is how long that decision actually stood. Judging a
    sweep stepping two days over a five-day window would score a position the
    strategy never held. The configured value still governs live runs.
    """
    # run_id becomes a path segment, so it is validated like a ticker: an
    # absolute or dotted value would otherwise place the run outside results_dir.
    run_id = safe_ticker_component(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["results_dir"]) / "backtest" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "results_dir": str(run_dir),
                  "memory_log_path": str(run_dir / "trading_memory.md"),
                  "cache_tool_fetches": cache_fetches,
                  "canonical_tool_windows": canonical_windows}
    holding = _grid_holding_days(dates, asset_type)
    if holding is not None:
        run_config["holding_period_days"] = holding

    graph = TradingAgentsGraph(selected_analysts, config=run_config)
    result = BacktestResult(run_id=run_id, log_path=Path(run_config["memory_log_path"]))
    done = {(e["ticker"], e["date"]) for e in graph.memory_log.load_entries()}

    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)
    try:
        for ticker in tickers:
            for date in dates:
                if (ticker, date) in done:
                    result.skipped += 1
                    continue
                try:
                    with collector.cell(ticker, date):
                        graph.propagate(ticker, date, asset_type, portfolio=portfolio)
                    result.cells_run += 1
                except Exception as exc:  # one unreachable vendor must not end the sweep
                    logger.warning("Backtest cell %s %s failed: %s", ticker, date, exc)
                    result.failures.append((ticker, date, str(exc)))

        # Settlement runs at the start of the next run for a ticker, so each ticker's
        # last cell would stay pending without this pass.
        for ticker in tickers:
            try:
                graph.settle_pending(ticker)
            except Exception as exc:  # reflection calls an LLM; one failure is not the sweep's
                logger.warning("Settling %s failed: %s", ticker, exc)
                result.settlement_failures.append((ticker, str(exc)))
    finally:
        fetch_issues.set_collector(None)
    result.fetch_issues = collector.issues

    # One metric for the whole sweep: a table mixing alpha and raw rows would
    # put two different questions in one column. One instrument whose benchmark
    # says nothing about it takes the sweep with it.
    if any(graph.scoring_metric(t, asset_type) == "raw" for t in tickers):
        result.metric = "raw"

    entries = graph.memory_log.load_entries()
    result.curves = build_curves(entries, run_config)
    try:
        result.report_path = write_html_report(
            result, summarize(graph.memory_log, result.metric), entries, run_config,
        )
    except Exception as exc:  # a report is a view of the sweep, never its point
        logger.warning("Could not write the backtest report: %s", exc)
    return result


def summarize(memory_log: TradingMemoryLog, metric: str = "alpha") -> BacktestSummary:
    """Score the settled decisions in a log, by rating.

    ``metric`` selects the outcome scored: ``"alpha"`` against the instrument's
    benchmark, or ``"raw"`` where that benchmark says nothing about it (see
    ``TradingAgentsGraph.scoring_metric``).
    """
    entries = memory_log.load_entries()
    # A decision with no readable rating has no direction, so it can neither
    # count for nor against the system; it is reported as unscored instead.
    resolved = [(e, _outcome(e, metric)) for e in entries
                if not e["pending"] and e["rating"] != RATING_REVIEW]
    resolved = [(e, a) for e, a in resolved if a is not None]
    by_rating: dict[str, RatingScore] = {}
    for rating in dict.fromkeys(e["rating"] for e, _ in resolved):
        values = [a for e, a in resolved if e["rating"] == rating]
        direction = _DIRECTION.get(rating, 0)
        by_rating[rating] = RatingScore(
            count=len(values),
            hit_rate=(sum(v * direction > 0 for v in values) / len(values)) if direction else None,
            mean_value=sum(values) / len(values),
        )
    unscored = sum(1 for e in entries if e["rating"] == RATING_REVIEW)
    # Report the window the outcomes were actually measured over, from the log.
    windows = {f"{e['holding'][:-1]} trading days" for e, _ in resolved
               if (e.get("holding") or "").endswith("d")}
    return BacktestSummary(resolved=len(resolved),
                           pending=len(entries) - len(resolved) - unscored,
                           by_rating=by_rating, unscored=unscored,
                           holding=", ".join(sorted(windows)) or "the configured window",
                           metric=metric)
