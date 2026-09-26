"""Backtesting: many single-shot decisions, scored by the decision log.

A run already records its rating and later settles it with realized and alpha
return against the regional benchmark. A backtest is that machinery over a grid
of tickers and dates, aggregated. It evaluates decision quality; it does not
simulate a portfolio, so there is no execution, no fees and no equity curve.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.backtest import iter_grid, run_backtest, summarize

DECISION = "Rating: Buy\n\nbuy it"


@pytest.mark.unit
def test_grid_spacing_and_canonical_dates():
    assert iter_grid("2026-01-05", "2026-01-20", every_n_days=7) == ["2026-01-05", "2026-01-12", "2026-01-19"]


@pytest.mark.unit
def test_grid_stops_at_today(monkeypatch):
    import tradingagents.backtest as bt

    monkeypatch.setattr(bt, "get_current_date", lambda: "2026-01-10")
    assert iter_grid("2026-01-05", "2026-02-20", every_n_days=5) == ["2026-01-05", "2026-01-10"]


@pytest.mark.unit
def test_grid_rejects_a_non_canonical_date():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        iter_grid("2026-1-5", "2026-01-20")


class _FakeGraph:
    """Stands in for TradingAgentsGraph, writing to the log the harness gave it."""

    instances: list = []
    fail_on: set = set()

    def __init__(self, selected_analysts=None, config=None, **kw):
        self.analysts = list(selected_analysts) if selected_analysts else None
        self.config = config
        self.memory_log = TradingMemoryLog(config)
        self.calls = []
        self.settled = []
        _FakeGraph.instances.append(self)

    def propagate(self, ticker, trade_date, asset_type="stock", portfolio=None):
        self.calls.append((ticker, trade_date))
        if (ticker, trade_date) in _FakeGraph.fail_on:
            raise RuntimeError("vendor exploded")
        self.memory_log.store_decision(ticker, trade_date, DECISION)
        return {"final_trade_decision": DECISION}, "Buy"

    def settle_pending(self, ticker):
        self.settled.append(ticker)

    def scoring_metric(self, ticker, asset_type="stock"):
        return "alpha"


@pytest.fixture(autouse=True)
def _fake_graph(monkeypatch, tmp_path):
    import tradingagents.backtest as bt

    _FakeGraph.instances = []
    _FakeGraph.fail_on = set()
    monkeypatch.setattr(bt, "TradingAgentsGraph", _FakeGraph)
    return _FakeGraph


def _config(tmp_path):
    return {"results_dir": str(tmp_path / "results"),
            "memory_log_path": str(tmp_path / "live_trading_memory.md")}


@pytest.mark.unit
def test_the_live_decision_log_is_never_written(tmp_path):
    config = _config(tmp_path)
    result = run_backtest(["NVDA"], ["2026-01-05", "2026-01-12"], config)

    assert not (tmp_path / "live_trading_memory.md").exists()
    assert result.log_path.exists() and result.cells_run == 2


@pytest.mark.unit
def test_a_cell_already_in_the_log_is_not_run_again(tmp_path):
    config = _config(tmp_path)
    first = run_backtest(["NVDA"], ["2026-01-05"], config)

    again = run_backtest(["NVDA"], ["2026-01-05", "2026-01-12"], config, run_id=first.run_id)

    assert again.cells_run == 1 and again.skipped == 1
    assert _FakeGraph.instances[-1].calls == [("NVDA", "2026-01-12")]


@pytest.mark.unit
def test_every_ticker_is_settled_after_the_grid(tmp_path):
    """Settlement runs at the start of the next same-ticker run, so the last
    date of each ticker would stay pending without an explicit pass."""
    run_backtest(["NVDA", "AAPL"], ["2026-01-05", "2026-01-12"], _config(tmp_path))
    assert sorted(_FakeGraph.instances[-1].settled) == ["AAPL", "NVDA"]


@pytest.mark.unit
def test_a_failed_cell_does_not_abort_the_sweep(tmp_path):
    _FakeGraph.fail_on = {("NVDA", "2026-01-05")}
    result = run_backtest(["NVDA"], ["2026-01-05", "2026-01-12"], _config(tmp_path))

    assert result.cells_run == 1
    assert result.failures == [("NVDA", "2026-01-05", "vendor exploded")]


# --- reading the result ------------------------------------------------------

def _log_with(tmp_path, rows):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    for ticker, date, decision, outcome in rows:
        log.store_decision(ticker, date, decision)
        if outcome is not None:
            log.update_with_outcome(ticker, date, outcome[0], outcome[1], 5, "note", "2026-02-01")
    return log


@pytest.mark.unit
def test_summary_scores_resolved_cells_and_keeps_pending_out_of_the_average(tmp_path):
    log = _log_with(tmp_path, [
        ("NVDA", "2026-01-05", "Rating: Buy\n\nx", (0.10, 0.04)),
        ("NVDA", "2026-01-12", "Rating: Buy\n\nx", (-0.02, -0.02)),
        ("AAPL", "2026-01-05", "Rating: Sell\n\nx", None),
    ])

    summary = summarize(log)

    assert summary.resolved == 2 and summary.pending == 1
    buys = summary.by_rating["Buy"]
    assert buys.count == 2 and buys.hit_rate == 0.5 and round(buys.mean_value, 4) == 0.01
    assert "Sell" not in summary.by_rating  # unsettled: nothing to score yet


@pytest.mark.unit
def test_summary_states_what_it_cannot_prove(tmp_path):
    text = summarize(_log_with(tmp_path, [("NVDA", "2026-01-05", DECISION, (0.1, 0.05))])).render()
    assert "not archived" in text
    assert "one" in text.lower() and "sampl" in text.lower()


@pytest.mark.unit
def test_the_analyst_set_under_test_is_the_one_that_runs(tmp_path):
    """A backtest of a two-analyst setup must not silently run four."""
    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), selected_analysts=["market", "news"])
    assert _FakeGraph.instances[-1].analysts == ["market", "news"]


@pytest.mark.unit
def test_a_run_id_cannot_escape_the_results_directory(tmp_path):
    """run_id becomes a path segment, so it is validated like a ticker is."""
    with pytest.raises(ValueError):
        run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), run_id="../../escaped")
    with pytest.raises(ValueError):
        run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), run_id="/etc/cron.d/x")


@pytest.mark.unit
def test_a_failed_settlement_does_not_lose_the_remaining_tickers(tmp_path, monkeypatch):
    """Settlement reflects with an LLM, so it can fail; the sweep still returns
    its result and every other ticker still gets settled."""
    settled = []

    def _settle(self, ticker):
        if ticker == "NVDA":
            raise RuntimeError("reflector timed out")
        settled.append(ticker)

    monkeypatch.setattr(_FakeGraph, "settle_pending", _settle, raising=False)
    result = run_backtest(["NVDA", "AAPL"], ["2026-01-05"], _config(tmp_path))

    assert result.cells_run == 2
    assert settled == ["AAPL"]
    assert result.settlement_failures == [("NVDA", "reflector timed out")]


@pytest.mark.unit
def test_pending_note_appears_only_when_something_is_pending(tmp_path):
    settled = [("NVDA", "2026-01-05", DECISION, (0.1, 0.05))]
    assert "Pending" not in summarize(_log_with(tmp_path, settled)).render()
    assert "Pending" in summarize(_log_with(tmp_path, settled + [("AAPL", "2026-01-05", DECISION, None)])).render()


# --- scoring reads the direction the rating claimed ---------------------------

def _scored(tmp_path, rows):
    log = _log_with(tmp_path, rows)
    return summarize(log).by_rating


@pytest.mark.unit
def test_a_bearish_call_that_fell_counts_as_right(tmp_path):
    """Alpha below the benchmark is the outcome a Sell predicted; scoring it as
    a miss reported the system as wrong exactly when it was right."""
    scores = _scored(tmp_path, [
        ("NVDA", "2026-01-05", "**Rating**: Sell\n\nx", (-0.08, -0.05)),
        ("AAPL", "2026-01-05", "**Rating**: Underweight\n\nx", (-0.03, -0.02)),
    ])
    assert scores["Sell"].hit_rate == 1.0
    assert scores["Underweight"].hit_rate == 1.0


@pytest.mark.unit
def test_a_bearish_call_that_rose_counts_as_wrong(tmp_path):
    scores = _scored(tmp_path, [("NVDA", "2026-01-05", "**Rating**: Sell\n\nx", (0.08, 0.05))])
    assert scores["Sell"].hit_rate == 0.0


@pytest.mark.unit
def test_a_bullish_call_is_scored_the_same_way_as_before(tmp_path):
    scores = _scored(tmp_path, [
        ("NVDA", "2026-01-05", "**Rating**: Buy\n\nx", (0.10, 0.04)),
        ("AAPL", "2026-01-05", "**Rating**: Buy\n\nx", (-0.02, -0.02)),
    ])
    assert scores["Buy"].hit_rate == 0.5


@pytest.mark.unit
def test_hold_claims_no_direction_so_it_gets_no_hit_rate(tmp_path):
    scores = _scored(tmp_path, [("NVDA", "2026-01-05", "**Rating**: Hold\n\nx", (0.01, 0.005))])
    assert scores["Hold"].hit_rate is None
    assert scores["Hold"].mean_value == 0.005


@pytest.mark.unit
def test_the_report_names_the_window_the_scores_cover(tmp_path):
    text = summarize(_log_with(tmp_path, [
        ("NVDA", "2026-01-05", "**Rating**: Buy\n\nx", (0.1, 0.05))])).render()
    assert "5" in text and "day" in text.lower()
    assert "Hold" not in text or "no direction" in text.lower()


@pytest.mark.unit
def test_the_window_reported_is_the_one_the_outcomes_used(tmp_path):
    """The log records the window each outcome was measured over; the summary
    must not claim a different one."""
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    log.store_decision("NVDA", "2026-01-05", "**Rating**: Buy\n\nx")
    log.update_with_outcome("NVDA", "2026-01-05", 0.1, 0.04, 21, "note", "2026-02-01")

    assert "21 trading days" in summarize(log).render()


@pytest.mark.unit
def test_a_sweep_pins_its_inputs_but_leaves_the_defaults_alone(tmp_path):
    """A sweep is an experiment: same fetched data, same window per cell. A
    single live run is the product and keeps both freedoms (defaults are off)."""
    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path))
    config = _FakeGraph.instances[-1].config
    assert config["cache_tool_fetches"] is True
    assert config["canonical_tool_windows"] is True


@pytest.mark.unit
def test_a_sweep_can_opt_out_of_both(tmp_path):
    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path),
                 cache_fetches=False, canonical_windows=False)
    config = _FakeGraph.instances[-1].config
    assert config["cache_tool_fetches"] is False
    assert config["canonical_tool_windows"] is False


@pytest.mark.unit
@pytest.mark.parametrize("dates, asset_type, expected", [
    (["2026-01-05", "2026-01-06", "2026-01-07"], "crypto", 1),   # ogni giorno
    (["2026-01-05", "2026-01-07", "2026-01-09"], "crypto", 2),   # a giorni alterni
    (["2026-01-05", "2026-01-10", "2026-01-15"], "crypto", 5),
    (["2026-01-05", "2026-01-12", "2026-01-19"], "stock", 5),    # 7 solari = 5 sedute
    (["2026-01-05", "2026-01-06"], "stock", 1),                  # mai sotto una barra
])
def test_the_horizon_follows_the_grid(tmp_path, dates, asset_type, expected):
    """A decision stands until the next one replaces it, so the gap between
    cells is how long it actually held — judging it over any other window
    scores a position the strategy never took."""
    run_backtest(["NVDA"], dates, _config(tmp_path), asset_type=asset_type)
    assert _FakeGraph.instances[-1].config["holding_period_days"] == expected


@pytest.mark.unit
def test_an_uneven_grid_keeps_the_configured_horizon(tmp_path):
    config = {**_config(tmp_path), "holding_period_days": 9}
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06", "2026-01-20"], config)
    assert _FakeGraph.instances[-1].config["holding_period_days"] == 9


@pytest.mark.unit
def test_a_sweep_never_learns_from_its_own_decisions(tmp_path):
    """Each model would learn from its own earlier calls, so two sweeps would
    feed their Portfolio Managers different lessons on the same cell."""
    config = {**_config(tmp_path), "learn_from_past_decisions": True}
    run_backtest(["NVDA"], ["2026-01-05"], config)
    assert _FakeGraph.instances[-1].config["learn_from_past_decisions"] is False


def _manifest(tmp_path, run_id="m"):
    import json

    path = tmp_path / "results" / "backtest" / run_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_the_conditions_of_a_sweep_are_written_beside_its_log(tmp_path):
    config = {**_config(tmp_path), "llm_provider": "openai", "deep_think_llm": "deep-1",
              "quick_think_llm": "quick-1", "max_tokens": "8192"}
    result = run_backtest(["BTC-USD"], ["2026-01-05", "2026-01-06"], config,
                          asset_type="crypto", selected_analysts=["market"], run_id="m")
    stored = _manifest(tmp_path)
    assert stored["model"]["deep_think_llm"] == "deep-1"
    assert stored["model"]["max_tokens"] == 8192  # as forwarded, not as the env spelled it
    assert stored["grid"] == {"tickers": ["BTC-USD"], "asset_type": "crypto", "first": "2026-01-05",
                              "last": "2026-01-06", "cells": 2, "every_days": 1}
    assert stored["pipeline"]["analysts"] == ["market"]
    assert stored["pipeline"]["holding_period_days"] == 1
    assert stored["sweep"]["learn_from_past_decisions"] is False
    assert stored["started_at"]
    assert result.manifest["model"] == stored["model"]
    assert result.manifest_mismatch == []


@pytest.mark.unit
def test_resuming_under_another_model_is_flagged(tmp_path):
    dates = ["2026-01-05", "2026-01-06"]
    run_backtest(["NVDA"], dates, {**_config(tmp_path), "deep_think_llm": "a"}, run_id="m")
    result = run_backtest(["NVDA"], [*dates, "2026-01-07"],
                          {**_config(tmp_path), "deep_think_llm": "b"}, run_id="m")
    assert result.manifest_mismatch == ["model"]
    assert _manifest(tmp_path)["model"]["deep_think_llm"] == "a"  # the original stays the reference


@pytest.mark.unit
def test_extending_the_grid_at_the_same_step_is_not_a_change_of_conditions(tmp_path):
    dates = ["2026-01-05", "2026-01-06"]
    run_backtest(["NVDA"], dates, _config(tmp_path), run_id="m")
    result = run_backtest(["NVDA"], [*dates, "2026-01-07"], _config(tmp_path), run_id="m")
    assert result.manifest_mismatch == []
    assert (result.skipped, result.cells_run) == (2, 1)


@pytest.mark.unit
def test_changing_the_step_changes_the_horizon_and_is_flagged(tmp_path):
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path),
                 asset_type="crypto", run_id="m")
    result = run_backtest(["NVDA"], ["2026-01-05", "2026-01-07"], _config(tmp_path),
                          asset_type="crypto", run_id="m")
    assert result.manifest_mismatch == ["pipeline"]


@pytest.mark.unit
def test_cells_logged_before_any_manifest_stay_flagged(tmp_path):
    log_path = tmp_path / "results" / "backtest" / "m" / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(log_path)}).store_decision("NVDA", "2026-01-05", DECISION)
    first = run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="m")
    again = run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="m")
    assert first.manifest_mismatch == ["unrecorded"]
    assert again.manifest_mismatch == ["unrecorded"]
    assert _manifest(tmp_path)["cells_before_manifest"] == 1


@pytest.mark.unit
def test_the_ollama_server_settings_are_read_from_its_own_log(tmp_path, monkeypatch):
    from tradingagents.backtest import _ollama_server_settings

    log = tmp_path / "Ollama" / "server.log"
    log.parent.mkdir()
    log.write_text(
        'time=1 msg="server config" env="map[OLLAMA_FLASH_ATTENTION:false OLLAMA_KV_CACHE_TYPE:]"\n'
        'time=2 msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0 X:y]"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "f16")  # a terminal's stale view must not win
    settings = _ollama_server_settings({"backend_url": "http://localhost:11434/v1"})
    assert settings == {"kv_cache_type": "q8_0", "flash_attention": "true", "read_from": "server log"}

    remote = _ollama_server_settings({"backend_url": "http://gpu-box:11434/v1"})
    assert remote["read_from"] == "client environment"
    assert remote["kv_cache_type"] == "f16"


@pytest.mark.unit
def test_an_ollama_model_name_is_pinned_to_its_weights_and_context(monkeypatch):
    import io
    import json
    import urllib.request

    from tradingagents.backtest import _ollama_models

    replies = {
        "/api/tags": {"models": [{"name": "fin-r1:latest", "digest": "0a7ec91547b6ffff"}]},
        "/api/show": {"parameters": 'num_ctx                        32768\nstop "<|im_end|>"',
                      "details": {"quantization_level": "Q5_K_M", "parameter_size": "7.6B"}},
    }

    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(replies[request.full_url.split("11434")[1]]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    models = _ollama_models({"backend_url": "http://localhost:11434/v1",
                             "deep_think_llm": "fin-r1:latest", "quick_think_llm": "fin-r1:latest"})
    assert models == {"fin-r1:latest": {"digest": "0a7ec91547b6", "num_ctx": 32768,
                                        "quantization": "Q5_K_M", "parameters": "7.6B"}}


@pytest.mark.unit
def test_an_unreachable_ollama_does_not_stop_the_sweep(monkeypatch):
    import urllib.request

    from tradingagents.backtest import _ollama_models

    def refuse(request, timeout):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert _ollama_models({"backend_url": "http://localhost:11434/v1", "deep_think_llm": "x"}) is None


@pytest.fixture
def _curves(monkeypatch):
    """The progress curve without the network: record what it was asked for."""
    import tradingagents.backtest as bt

    asked = []
    monkeypatch.setattr(bt, "build_curve", lambda entries, ticker, config, until=None:
                        asked.append((ticker, until, len(entries))) or f"curve {ticker} {until}")
    return asked


@pytest.mark.unit
def test_every_cell_run_is_reported_as_it_finishes(tmp_path, _curves):
    seen = []
    run_backtest(["NVDA", "AAPL"], ["2026-01-05", "2026-01-06"], _config(tmp_path), on_cell=seen.append)
    assert [(p.ticker, p.date, p.done, p.total) for p in seen] == [
        ("NVDA", "2026-01-05", 1, 4), ("NVDA", "2026-01-06", 2, 4),
        ("AAPL", "2026-01-05", 3, 4), ("AAPL", "2026-01-06", 4, 4),
    ]
    assert all(p.rating == "Buy" and p.error is None for p in seen)
    assert seen[1].curve == "curve NVDA 2026-01-06"  # read up to the cell just run
    assert _curves[1] == ("NVDA", "2026-01-06", 2)   # with the decisions logged so far
    assert seen[-1].elapsed >= seen[0].elapsed >= 0


@pytest.mark.unit
def test_a_resumed_sweep_counts_only_what_is_left(tmp_path, _curves):
    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), run_id="r")
    seen = []
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="r", on_cell=seen.append)
    assert [(p.date, p.done, p.total) for p in seen] == [("2026-01-06", 1, 1)]


@pytest.mark.unit
def test_a_failed_cell_is_reported_with_its_error(tmp_path, _curves):
    _FakeGraph.fail_on = {("NVDA", "2026-01-06")}
    seen = []
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), on_cell=seen.append)
    assert seen[1].error == "vendor exploded" and seen[1].rating is None
    assert seen[1].done == 2


@pytest.mark.unit
def test_the_time_left_is_estimated_from_the_cells_so_far():
    from tradingagents.backtest import CellProgress

    halfway = CellProgress("NVDA", "2026-01-05", done=2, total=6, seconds=50, elapsed=100)
    assert halfway.remaining_seconds == 200


@pytest.mark.unit
def test_a_broken_display_does_not_cost_the_sweep(tmp_path, _curves):
    def explode(progress):
        raise RuntimeError("terminal gone")

    result = run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), on_cell=explode)
    assert result.cells_run == 2 and result.failures == []


@pytest.mark.unit
def test_unreachable_prices_leave_the_cell_reported_without_a_curve(tmp_path, monkeypatch):
    import tradingagents.backtest as bt

    def offline(*args, **kwargs):
        raise ConnectionError("no route")

    monkeypatch.setattr(bt, "build_curve", offline)
    seen = []
    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), on_cell=seen.append)
    assert len(seen) == 1 and seen[0].curve is None


@pytest.fixture
def _offline_curves(monkeypatch):
    import tradingagents.backtest as bt

    monkeypatch.setattr(bt, "build_curves", lambda entries, config: [])
    monkeypatch.setattr(bt, "build_curve", lambda *a, **k: None)


def _issue_on(monkeypatch, cells):
    """Make the fake graph hit a fetch issue in the given cells."""
    from tradingagents.dataflows import fetch_issues

    original = _FakeGraph.propagate

    def propagate(self, ticker, trade_date, *args, **kwargs):
        if (ticker, trade_date) in cells:
            fetch_issues.record("get_news", "no_coverage", f"nothing for {trade_date}", (ticker,))
        return original(self, ticker, trade_date, *args, **kwargs)

    monkeypatch.setattr(_FakeGraph, "propagate", propagate)


def _sweep_state(tmp_path, run_id="s"):
    import json

    return json.loads((tmp_path / "results" / "backtest" / run_id / "sweep.json").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_what_a_sweep_recorded_outlives_the_process(tmp_path, monkeypatch, _offline_curves):
    _issue_on(monkeypatch, {("NVDA", "2026-01-05")})
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="s")
    state = _sweep_state(tmp_path)
    assert [(i["date"], i["reason"]) for i in state["fetch_issues"]] == [("2026-01-05", "no_coverage")]
    assert state["metric"] == "alpha" and state["failures"] == []


@pytest.mark.unit
def test_a_resumed_sweep_keeps_every_session_s_issues(tmp_path, monkeypatch, _offline_curves):
    import tradingagents.backtest as bt

    _issue_on(monkeypatch, {("NVDA", "2026-01-05"), ("NVDA", "2026-01-06")})
    written = []
    monkeypatch.setattr(bt, "write_html_report", lambda result, *a: written.append(result) or None)
    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), run_id="s")
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="s")
    assert [i["date"] for i in _sweep_state(tmp_path)["fetch_issues"]] == ["2026-01-05", "2026-01-06"]
    assert len(written[-1].fetch_issues) == 2   # the report reads the whole sweep


@pytest.mark.unit
def test_a_failed_cell_leaves_the_record_once_a_later_session_logs_it(tmp_path, _offline_curves):
    _FakeGraph.fail_on = {("NVDA", "2026-01-06")}
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="s")
    assert [f[1] for f in _sweep_state(tmp_path)["failures"]] == ["2026-01-06"]
    _FakeGraph.fail_on = set()
    run_backtest(["NVDA"], ["2026-01-05", "2026-01-06"], _config(tmp_path), run_id="s")
    assert _sweep_state(tmp_path)["failures"] == []


@pytest.mark.unit
def test_a_report_is_written_again_without_running_a_cell(tmp_path, monkeypatch, _offline_curves):
    import tradingagents.backtest as bt
    from tradingagents.backtest import rebuild_report

    _issue_on(monkeypatch, {("BTC-USD", "2026-01-05")})
    run_backtest(["BTC-USD"], ["2026-01-05", "2026-01-06"], _config(tmp_path),
                 asset_type="crypto", run_id="s")
    run_dir = tmp_path / "results" / "backtest" / "s"
    manifest_before = (run_dir / "manifest.json").read_text(encoding="utf-8")
    (run_dir / "report.html").unlink()
    graphs_before = len(_FakeGraph.instances)

    readings = []
    monkeypatch.setattr(bt, "build_curves", lambda entries, config: readings.append(config) or [])
    result = rebuild_report("s", _config(tmp_path))

    assert result.report_path.exists()
    assert len(_FakeGraph.instances) == graphs_before           # no graph, no cell run
    assert (run_dir / "manifest.json").read_text(encoding="utf-8") == manifest_before
    assert result.metric == _sweep_state(tmp_path)["metric"]    # as scored, not re-inferred
    assert len(result.fetch_issues) == 1
    assert readings[0]["holding_period_days"] == 1              # the window the sweep ran with
    assert "no_coverage" in result.report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_there_is_no_report_to_rebuild_for_a_sweep_that_never_ran(tmp_path):
    from tradingagents.backtest import rebuild_report

    with pytest.raises(FileNotFoundError, match="No sweep 'ghost'"):
        rebuild_report("ghost", _config(tmp_path))


@pytest.mark.unit
def test_a_report_can_read_the_same_decisions_as_orders(tmp_path, monkeypatch, _offline_curves):
    import tradingagents.backtest as bt
    from tradingagents.backtest import rebuild_report

    run_backtest(["BTC-USD"], ["2026-01-05", "2026-01-06"], _config(tmp_path),
                 asset_type="crypto", run_id="s")
    run_dir = tmp_path / "results" / "backtest" / "s"
    own_report = (run_dir / "report.html").read_text(encoding="utf-8")
    readings = []
    monkeypatch.setattr(bt, "build_curves", lambda entries, config: readings.append(config) or [])

    orders = {"Buy": 1.0, "Overweight": 0.5, "Underweight": -0.5, "Sell": -1.0}
    result = rebuild_report("s", _config(tmp_path), trades=orders)

    assert result.report_path == run_dir / "report_trades.html"
    assert readings[0]["strategy_trades"] == orders
    assert (run_dir / "report.html").read_text(encoding="utf-8") == own_report  # left as it was
    assert "ordini: Buy compra il 100% del patrimonio" in result.report_path.read_text(encoding="utf-8")


@pytest.mark.unit
@pytest.mark.parametrize("name", ["../elsewhere.html", "report.txt", "sub/report.html"])
def test_a_rebuilt_report_stays_beside_its_log(tmp_path, _offline_curves, name):
    from tradingagents.backtest import rebuild_report

    run_backtest(["NVDA"], ["2026-01-05"], _config(tmp_path), run_id="s")
    with pytest.raises(ValueError, match="file name ending in .html"):
        rebuild_report("s", _config(tmp_path), filename=name)
