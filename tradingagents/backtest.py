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

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATING_REVIEW
from tradingagents.backtest_report import write_html_report
from tradingagents.dataflows import fetch_issues
from tradingagents.dataflows.utils import get_current_date, safe_ticker_component
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.strategy_curve import StrategyCurve, build_curve, build_curves

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
    # How this sweep was produced (see _run_manifest), and the sections that
    # differ from the manifest the run was started under, when it was resumed.
    manifest: dict = field(default_factory=dict)
    manifest_mismatch: list[str] = field(default_factory=list)


@dataclass
class CellProgress:
    """A cell of a running sweep has finished: what a live display shows of it.

    A sweep is silent while the graph runs, and one can take a day, so the
    caller of run_backtest() gets one of these per cell it runs. ``done`` and
    ``total`` count only the cells this invocation runs: cells already in a
    resumed log are neither.
    """

    ticker: str
    date: str
    done: int
    total: int
    seconds: float          # this cell's wall time
    elapsed: float          # since this invocation started
    rating: str | None = None
    error: str | None = None
    # This ticker's decisions so far, read up to this cell's close.
    curve: StrategyCurve | None = None

    @property
    def remaining_seconds(self) -> float:
        return self.elapsed / self.done * (self.total - self.done) if self.done else 0.0


def _report_cell(on_cell: Callable[[CellProgress], None], graph, config: dict,
                 progress: CellProgress) -> None:
    try:
        progress.curve = build_curve(graph.memory_log.load_entries(), progress.ticker,
                                     config, until=progress.date)
    except Exception as exc:  # prices can be briefly unreachable; the cell still counts
        logger.debug("No curve for %s up to %s: %s", progress.ticker, progress.date, exc)
    try:
        on_cell(progress)
    except Exception as exc:  # a display must never cost a sweep that is hours in
        logger.warning("Progress display failed: %s", exc)


_REPO = Path(__file__).resolve().parent.parent

# The manifest sections that decide what a cell computes. A resumed sweep is
# checked against these only: extending the grid or re-reading the decisions
# under other strategy weights leaves every existing cell valid.
_CELL_SECTIONS = ("code", "model", "server", "pipeline", "sweep")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _code_state() -> dict:
    """The code a sweep ran, as precisely as git can say.

    Tree hashes of the two packages rather than the commit: a commit touching
    only docs or tests changes nothing a cell computes, and must not read as a
    different configuration on resume. Uncommitted edits are fingerprinted by
    their content, since no hash in the history describes them.
    """
    if not (_REPO / ".git").exists():  # an installed package, not a checkout
        return {}
    dirty = "\n".join(filter(None, (
        _git("status", "--porcelain", "--", "tradingagents", "cli"),
        _git("diff", "HEAD", "--", "tradingagents", "cli"),
    )))
    return {
        "tradingagents": _git("rev-parse", "HEAD:./tradingagents"),
        "cli": _git("rev-parse", "HEAD:./cli"),
        "uncommitted": hashlib.sha256(dirty.encode("utf-8")).hexdigest()[:12] if dirty else None,
    }


def _ollama_models(config: dict) -> dict | None:
    """What each Ollama model name pointed at when the sweep ran.

    A model name is a label: re-creating it from another Modelfile keeps the
    name and changes the weights or the context window, and either changes the
    answers. The digest and ``num_ctx`` pin what the name meant.
    """
    from urllib.request import Request, urlopen

    base = (config.get("backend_url") or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1")
    root = base.rstrip("/").removesuffix("/v1")
    if not root.startswith(("http://", "https://")):
        return None

    def call(path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(root + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    try:
        digests = {m["name"]: m.get("digest") for m in call("/api/tags").get("models", [])}
        models = {}
        for name in sorted({config.get("deep_think_llm"), config.get("quick_think_llm")} - {None}):
            shown = call("/api/show", {"model": name})
            num_ctx = None
            for line in (shown.get("parameters") or "").splitlines():
                key, _, value = line.strip().partition(" ")
                if key == "num_ctx":
                    num_ctx = int(value.strip())
            details = shown.get("details") or {}
            digest = digests.get(name) or digests.get(f"{name}:latest")
            models[name] = {
                "digest": digest[:12] if digest else None,
                "num_ctx": num_ctx,
                "quantization": details.get("quantization_level"),
                "parameters": details.get("parameter_size"),
            }
        return models
    except Exception as exc:  # a best-effort record; the sweep itself says if Ollama is down
        logger.warning("Could not read the Ollama model details from %s: %s", root, exc)
        return None


def _ollama_server_settings(config: dict) -> dict:
    """The KV-cache settings the Ollama server runs with.

    No API reports them, and this process's environment need not be the
    server's: a terminal opened before the variables were set sees none of
    them. The server prints its own configuration to its log when it starts,
    so a local server is read from there, and anything else falls back to the
    client's environment and says so.
    """
    from urllib.parse import urlparse

    base = (config.get("backend_url") or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1")
    logs = [Path.home() / ".ollama" / "logs" / "server.log"]
    if os.environ.get("LOCALAPPDATA"):
        logs.insert(0, Path(os.environ["LOCALAPPDATA"]) / "Ollama" / "server.log")
    if urlparse(base).hostname in ("localhost", "127.0.0.1", "::1"):
        for log in logs:
            try:
                started = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                           if 'msg="server config"' in line]
            except OSError:
                continue
            if started:
                def setting(name: str, line: str = started[-1]) -> str:
                    found = re.search(rf"\b{name}:(\S*)", line)
                    return (found.group(1) if found else "") or "default"
                return {"kv_cache_type": setting("OLLAMA_KV_CACHE_TYPE"),
                        "flash_attention": setting("OLLAMA_FLASH_ATTENTION"),
                        "read_from": "server log"}
    return {"kv_cache_type": os.environ.get("OLLAMA_KV_CACHE_TYPE") or "default",
            "flash_attention": os.environ.get("OLLAMA_FLASH_ATTENTION") or "default",
            "read_from": "client environment"}


def _number(value, kind):
    return None if value is None or value == "" else kind(value)


def _run_manifest(config: dict, tickers: list[str], dates: list[str], asset_type: str,
                  selected_analysts) -> dict:
    """Everything that decides what a sweep's numbers mean, in one record.

    Two sweeps are comparable only when this matches except for the model, so
    it is written beside the log and printed in the report.
    """
    gaps = {(datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days
            for a, b in zip(dates, dates[1:], strict=False)}
    manifest = {
        "commit": _git("rev-parse", "HEAD") if (_REPO / ".git").exists() else None,
        "code": _code_state(),
        "model": {
            "provider": config.get("llm_provider"),
            "deep_think_llm": config.get("deep_think_llm"),
            "quick_think_llm": config.get("quick_think_llm"),
            # As the graph forwards them: from the environment they arrive as
            # strings, and "8192" and 8192 must not read as two settings.
            "temperature": _number(config.get("temperature"), float),
            "max_tokens": _number(config.get("max_tokens"), int),
        },
        "server": None,
        "grid": {
            "tickers": list(tickers),
            "asset_type": asset_type,
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "cells": len(tickers) * len(dates),
            "every_days": gaps.pop() if len(gaps) == 1 else None,
        },
        "pipeline": {
            "analysts": list(selected_analysts),
            "max_debate_rounds": config.get("max_debate_rounds"),
            "max_risk_discuss_rounds": config.get("max_risk_discuss_rounds"),
            "holding_period_days": config.get("holding_period_days"),
            "data_vendors": config.get("data_vendors"),
            "tool_vendors": config.get("tool_vendors"),
        },
        "sweep": {
            "cache_tool_fetches": config.get("cache_tool_fetches"),
            "canonical_tool_windows": config.get("canonical_tool_windows"),
            "news_lookback_days": config.get("news_lookback_days"),
            "price_lookback_days": config.get("price_lookback_days"),
            "learn_from_past_decisions": config.get("learn_from_past_decisions"),
        },
        "strategy": {
            "positions": config.get("strategy_positions"),
            "cost_bps": config.get("strategy_cost_bps"),
        },
    }
    if config.get("llm_provider") == "ollama":
        manifest["server"] = {"models": _ollama_models(config), **_ollama_server_settings(config)}
    return manifest


def _record_manifest(run_dir: Path, manifest: dict, prior_cells: int = 0) -> list[str]:
    """Store a sweep's manifest, or on resume name what no longer matches.

    A resumed sweep keeps the cells it already has, so continuing it under
    other code, another model or another cache setting would mix two
    configurations in one log without a trace. The manifest the sweep started
    under stays the reference. Cells logged before any manifest existed are
    of unknown provenance, and stay reported as such.
    """
    path = run_dir / "manifest.json"
    if not path.exists():
        original = {**manifest, "started_at": datetime.now().isoformat(timespec="seconds")}
        if prior_cells:
            original["cells_before_manifest"] = prior_cells
        path.write_text(json.dumps(original, indent=2, default=str), encoding="utf-8")
    else:
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return []
    mismatch = ["unrecorded"] if original.get("cells_before_manifest") else []
    mismatch += [k for k in _CELL_SECTIONS
                 if json.dumps(original.get(k), sort_keys=True, default=str)
                 != json.dumps(manifest.get(k), sort_keys=True, default=str)]
    if mismatch:
        logger.warning(
            "Sweep %s mixes cells produced under different conditions (%s): they are "
            "not all comparable. Use a new --run-id to compare configurations.",
            run_dir.name, ", ".join(mismatch),
        )
    return mismatch


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
    on_cell: Callable[[CellProgress], None] | None = None,
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

    ``on_cell``, when given, is called after every cell this invocation runs,
    failed ones included, with a CellProgress carrying the ticker's strategy
    curve up to that date. It is a display hook: whatever it raises is logged
    and the sweep goes on.
    """
    # run_id becomes a path segment, so it is validated like a ticker: an
    # absolute or dotted value would otherwise place the run outside results_dir.
    run_id = safe_ticker_component(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["results_dir"]) / "backtest" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "results_dir": str(run_dir),
                  "memory_log_path": str(run_dir / "trading_memory.md"),
                  "cache_tool_fetches": cache_fetches,
                  "canonical_tool_windows": canonical_windows,
                  # Every cell decides from the same inputs: no lessons from the
                  # sweep's own earlier decisions, which differ model by model.
                  "learn_from_past_decisions": False}
    holding = _grid_holding_days(dates, asset_type)
    if holding is not None:
        run_config["holding_period_days"] = holding

    graph = TradingAgentsGraph(selected_analysts, config=run_config)
    result = BacktestResult(run_id=run_id, log_path=Path(run_config["memory_log_path"]))
    done = {(e["ticker"], e["date"]) for e in graph.memory_log.load_entries()}
    result.manifest = _run_manifest(run_config, tickers, dates, asset_type, selected_analysts)
    result.manifest_mismatch = _record_manifest(run_dir, result.manifest, len(done))

    total = sum((t, d) not in done for t in tickers for d in dates)
    started = time.monotonic()
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)
    try:
        for ticker in tickers:
            for date in dates:
                if (ticker, date) in done:
                    result.skipped += 1
                    continue
                cell_started = time.monotonic()
                rating = error = None
                try:
                    with collector.cell(ticker, date):
                        _, rating = graph.propagate(ticker, date, asset_type, portfolio=portfolio)
                    result.cells_run += 1
                except Exception as exc:  # one unreachable vendor must not end the sweep
                    logger.warning("Backtest cell %s %s failed: %s", ticker, date, exc)
                    result.failures.append((ticker, date, str(exc)))
                    error = str(exc)
                if on_cell is not None:
                    now = time.monotonic()
                    _report_cell(on_cell, graph, run_config, CellProgress(
                        ticker=ticker, date=date,
                        done=result.cells_run + len(result.failures), total=total,
                        seconds=now - cell_started, elapsed=now - started,
                        rating=rating, error=error,
                    ))

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
    state = _record_sweep(run_dir, result, {(e["ticker"], e["date"]) for e in entries})
    try:
        result.report_path = write_html_report(
            _whole_sweep(result, state), summarize(graph.memory_log, result.metric),
            entries, run_config,
        )
    except Exception as exc:  # a report is a view of the sweep, never its point
        logger.warning("Could not write the backtest report: %s", exc)
    return result


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _record_sweep(run_dir: Path, result: BacktestResult, logged: set[tuple[str, str]]) -> dict:
    """Keep what a sweep recorded besides its decisions, across every session.

    Fetch issues and failures lived only in memory: a resumed sweep's report
    showed the last session's alone, and no report could be written again
    once the process had exited. Issues accumulate; a failed cell is dropped
    once a later session has logged it.
    """
    path = run_dir / "sweep.json"
    prior = _read_json(path) or {}
    failures = {(t, d): reason for t, d, reason in prior.get("failures", [])}
    failures.update({(t, d): reason for t, d, reason in result.failures})
    state = {
        "metric": result.metric,
        "fetch_issues": prior.get("fetch_issues", []) + result.fetch_issues,
        "failures": [[t, d, reason] for (t, d), reason in failures.items() if (t, d) not in logged],
        # Settlement is retried by every session, so only the latest attempt counts.
        "settlement_failures": [list(f) for f in result.settlement_failures],
        "manifest_mismatch": sorted(set(prior.get("manifest_mismatch", []))
                                    | set(result.manifest_mismatch)),
    }
    try:
        path.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not record %s: %s", path, exc)
    return state


def _whole_sweep(result: BacktestResult, state: dict) -> BacktestResult:
    """This session's result, carrying what every session of the sweep recorded."""
    return replace(
        result,
        fetch_issues=state["fetch_issues"],
        failures=[tuple(f) for f in state["failures"]],
        settlement_failures=[tuple(f) for f in state["settlement_failures"]],
        manifest_mismatch=state["manifest_mismatch"],
    )


def rebuild_report(run_id: str, config: dict) -> BacktestResult:
    """Write a finished sweep's report again from what it left on disk.

    Nothing is re-run and nothing the sweep recorded is changed: its
    decisions, the manifest of the conditions they were made under, and its
    fetch issues are read back as they are. Only the reading is redone with
    the current code, under the holding window and strategy rule the sweep
    ran with, so two sweeps' reports can be brought to the same reading.
    """
    run_id = safe_ticker_component(run_id)
    run_dir = Path(config["results_dir"]) / "backtest" / run_id
    log_path = run_dir / "trading_memory.md"
    if not log_path.exists():
        raise FileNotFoundError(f"No sweep '{run_id}' under {run_dir.parent}")
    manifest = _read_json(run_dir / "manifest.json") or {}
    state = _read_json(run_dir / "sweep.json") or {}
    model = manifest.get("model") or {}
    pipeline = manifest.get("pipeline") or {}
    strategy = manifest.get("strategy") or {}
    reading = {
        **config,
        "llm_provider": model.get("provider", config.get("llm_provider")),
        "deep_think_llm": model.get("deep_think_llm", config.get("deep_think_llm")),
        "quick_think_llm": model.get("quick_think_llm", config.get("quick_think_llm")),
        "holding_period_days": pipeline.get("holding_period_days", config.get("holding_period_days")),
        "strategy_positions": strategy.get("positions", config.get("strategy_positions")),
        "strategy_cost_bps": strategy.get("cost_bps", config.get("strategy_cost_bps")),
    }
    memory_log = TradingMemoryLog({"memory_log_path": str(log_path)})
    entries = memory_log.load_entries()
    crypto = (manifest.get("grid") or {}).get("asset_type") == "crypto"
    mismatch = list(state.get("manifest_mismatch", []))
    if manifest.get("cells_before_manifest") and "unrecorded" not in mismatch:
        mismatch.insert(0, "unrecorded")
    result = BacktestResult(
        run_id=run_id, log_path=log_path, cells_run=len(entries),
        failures=[tuple(f) for f in state.get("failures", [])],
        settlement_failures=[tuple(f) for f in state.get("settlement_failures", [])],
        fetch_issues=state.get("fetch_issues", []),
        metric=state.get("metric") or ("raw" if crypto else "alpha"),
        manifest=manifest, manifest_mismatch=mismatch,
    )
    result.curves = build_curves(entries, reading)
    result.report_path = write_html_report(result, summarize(memory_log, result.metric),
                                           entries, reading)
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
