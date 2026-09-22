"""Canonical tool windows: an evaluation run pins every window length, a live
run leaves the model's own choice alone.

The point is comparability — a model that asks for 29 days of news simply has
more information than one that asks for 7, which would read as the better
reasoner. A single run is the product, not an experiment, so it keeps the
freedom to choose.
"""

import copy
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.date_window import canonical_span, canonical_window


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_config()
    yield
    _reset_config()


# --- the helpers ---------------------------------------------------------

@pytest.mark.unit
def test_a_live_run_keeps_the_models_span():
    assert canonical_span(29, 7) == 29


@pytest.mark.unit
def test_an_evaluation_run_pins_the_span():
    set_config({"canonical_tool_windows": True})
    assert canonical_span(29, 7) == 7


@pytest.mark.unit
def test_a_live_run_keeps_the_models_window():
    start, end = canonical_window("2026-08-10", "2026-09-08", "2026-09-08", 7)
    assert (start, end) == ("2026-08-10", "2026-09-08")


@pytest.mark.unit
def test_an_evaluation_run_pins_the_window_start():
    set_config({"canonical_tool_windows": True})
    start, end = canonical_window("2026-08-10", "2026-09-08", "2026-09-08", 7)
    assert (start, end) == ("2026-09-01", "2026-09-08")


@pytest.mark.unit
def test_two_models_asking_differently_get_the_same_window():
    """This is the whole point: the same cache key, hence the same input."""
    set_config({"canonical_tool_windows": True})
    a = canonical_window("2026-09-01", "2026-09-08", "2026-09-08", 7)
    b = canonical_window("2026-08-10", "2026-09-08", "2026-09-08", 7)
    assert a == b


@pytest.mark.unit
def test_pinning_never_defeats_the_look_ahead_clamp():
    """The end is still clamped to the run date, pinned window or not."""
    set_config({"canonical_tool_windows": True})
    _, end = canonical_window("2026-09-01", "2026-12-25", "2026-09-08", 7)
    assert end == "2026-09-08"


# --- the tools -----------------------------------------------------------

def _routed(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "route_to_vendor", lambda *a, **k: calls.append((a, k)) or "OK")
    return calls


@pytest.mark.unit
def test_news_window_is_pinned_only_for_an_evaluation_run(monkeypatch):
    import tradingagents.agents.utils.news_data_tools as news

    calls = _routed(monkeypatch, news)
    news.get_news.func("BTC-USD", "2026-08-10", "2026-09-08", "2026-09-08")
    assert calls[-1][0][2] == "2026-08-10"  # live: the model's own start

    set_config({"canonical_tool_windows": True, "news_lookback_days": 7})
    news.get_news.func("BTC-USD", "2026-08-10", "2026-09-08", "2026-09-08")
    assert calls[-1][0][2] == "2026-09-01"  # sweep: pinned to the configured span


@pytest.mark.unit
def test_indicator_lookback_is_pinned_only_for_an_evaluation_run(monkeypatch):
    import tradingagents.agents.utils.technical_indicators_tools as ind

    calls = _routed(monkeypatch, ind)
    ind.get_indicators.func("BTC-USD", "rsi", "2026-09-08", 120, "2026-09-08")
    assert calls[-1][0][4] == 120

    set_config({"canonical_tool_windows": True})
    ind.get_indicators.func("BTC-USD", "rsi", "2026-09-08", 120, "2026-09-08")
    assert calls[-1][0][4] == 30


@pytest.mark.unit
def test_global_news_span_and_limit_fall_back_to_config_for_a_sweep(monkeypatch):
    import tradingagents.agents.utils.news_data_tools as news

    calls = _routed(monkeypatch, news)
    news.get_global_news.func("2026-09-08", 30, 50, "2026-09-08")
    assert calls[-1][0][2:] == (30, 50)

    set_config({"canonical_tool_windows": True})
    news.get_global_news.func("2026-09-08", 30, 50, "2026-09-08")
    assert calls[-1][0][2:] == (None, None)  # None = use the configured default


@pytest.mark.unit
def test_the_verified_snapshot_lookback_is_pinned_for_a_sweep(monkeypatch):
    import tradingagents.agents.utils.market_data_validation_tools as tools

    builder = mock.Mock(return_value="SNAPSHOT")
    monkeypatch.setattr(tools, "build_verified_market_snapshot", builder)

    tools.get_verified_market_snapshot.func("BTC-USD", "2026-09-08", 5, "2026-09-08")
    assert builder.call_args[0][2] == 5

    set_config({"canonical_tool_windows": True})
    tools.get_verified_market_snapshot.func("BTC-USD", "2026-09-08", 5, "2026-09-08")
    assert builder.call_args[0][2] == 30


@pytest.mark.unit
def test_a_backtest_turns_it_on_but_it_stays_off_by_default():
    assert default_config.DEFAULT_CONFIG["canonical_tool_windows"] is False


@pytest.mark.unit
def test_the_prediction_market_limit_is_pinned_for_a_sweep(monkeypatch):
    """How many markets to return is a volume knob like the news limit; only the
    topic itself stays the model's to choose."""
    import tradingagents.agents.utils.prediction_markets_tools as pm

    calls = _routed(monkeypatch, pm)
    pm.get_prediction_markets.func("recession 2026", 5, "2026-09-08")
    assert calls[-1][0][2] == 5

    set_config({"canonical_tool_windows": True})
    pm.get_prediction_markets.func("recession 2026", 5, "2026-09-08")
    assert calls[-1][0][2] is None
