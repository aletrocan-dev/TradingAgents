"""A sweep does not learn from its own decisions; a live run still does.

Each model would learn from its own earlier calls, so two sweeps over the same
grid would feed their Portfolio Managers different lessons on the same cell, and
every reflection would be a model call spent on text nothing reads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _graph_with_pending(tmp_path, learn: bool):
    graph = MagicMock(spec=TradingAgentsGraph)
    graph.config = {"learn_from_past_decisions": learn, "holding_period_days": 1}
    graph.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "log.md")})
    graph.memory_log.store_decision("BTC-USD", "2026-01-07", "Rating: Buy\n\nbuy")
    graph._resolve_benchmark.return_value = "SPY"
    graph._fetch_returns.return_value = (0.1, 0.05, 1, "2026-01-08")
    graph.reflector = MagicMock()
    graph.reflector.reflect_on_final_decision.return_value = "a lesson"
    return graph


@pytest.mark.unit
def test_a_sweep_settles_outcomes_without_spending_a_call_on_reflection(tmp_path):
    graph = _graph_with_pending(tmp_path, learn=False)
    TradingAgentsGraph._resolve_pending_entries(graph, "BTC-USD")
    graph.reflector.reflect_on_final_decision.assert_not_called()
    (entry,) = graph.memory_log.load_entries()
    assert entry["pending"] is False and entry["raw"] == "+10.0%"
    assert entry["reflection"] == ""


@pytest.mark.unit
def test_a_live_run_still_reflects(tmp_path):
    graph = _graph_with_pending(tmp_path, learn=True)
    TradingAgentsGraph._resolve_pending_entries(graph, "BTC-USD")
    graph.reflector.reflect_on_final_decision.assert_called_once()
    assert graph.memory_log.load_entries()[0]["reflection"] == "a lesson"


@pytest.mark.unit
@pytest.mark.parametrize("learn, injected", [(False, False), (True, True)])
def test_lessons_reach_the_portfolio_manager_only_when_learning(tmp_path, learn, injected):
    graph = _graph_with_pending(tmp_path, learn=learn)
    graph.memory_log = MagicMock()
    graph.memory_log.get_past_context.return_value = "PAST LESSONS"
    graph.propagator = MagicMock()
    graph.resolve_instrument_context.return_value = ""
    graph._memory_as_of.return_value = None
    TradingAgentsGraph.create_run_state(graph, "BTC-USD", "2026-01-09")
    past = graph.propagator.create_initial_state.call_args.kwargs["past_context"]
    assert (past == "PAST LESSONS") is injected
    assert graph.memory_log.get_past_context.called is injected
