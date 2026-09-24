"""A reasoning model's scratchpad stays with the model that had it.

R1-family models write ``<think>...</think>`` before answering. Handed on, that
reads as the agent's position, hypotheses it rejected included. On a real sweep
it was a third of every debate history and half of the Research Manager's plan,
which pushed prompts past the context window.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.utils.structured import invoke_structured_or_freetext, strip_reasoning

THOUGHT = "<think>Maybe Sell? The RSI says otherwise, so maybe Buy.</think>\n"


@pytest.mark.unit
def test_the_answer_survives_and_the_deliberation_does_not():
    assert strip_reasoning(THOUGHT + "Rating: Hold") == "Rating: Hold"


@pytest.mark.unit
def test_a_thought_cut_off_by_the_token_cap_holds_no_answer():
    assert strip_reasoning("<think>still weighing Sell against") == ""


@pytest.mark.unit
def test_every_block_goes_and_case_does_not_matter():
    text = "<THINK>one</THINK>First. <think>two</think>Second."
    assert strip_reasoning(text) == "First. Second."


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, ["<think>x</think>"], {"a": 1}])
def test_anything_but_text_passes_through(value):
    assert strip_reasoning(value) is value


@pytest.mark.unit
def test_a_plain_answer_is_untouched():
    assert strip_reasoning("  Rating: Buy\n\nbecause  ") == "Rating: Buy\n\nbecause"


@pytest.mark.unit
def test_the_free_text_fallback_hands_on_the_answer_only():
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content=THOUGHT + "Rating: Buy")
    out = invoke_structured_or_freetext(None, plain, "prompt", str, "Trader")
    assert out == "Rating: Buy"


@pytest.mark.unit
def test_the_structured_path_hands_on_the_answer_only():
    structured = MagicMock()
    structured.invoke.return_value = object()
    out = invoke_structured_or_freetext(
        structured, MagicMock(), "prompt", lambda _: THOUGHT + "Rating: Sell", "Portfolio Manager",
    )
    assert out == "Rating: Sell"


def _thinking_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=THOUGHT + "my argument")
    return llm


_REPORTS = {
    "company_of_interest": "AAPL", "asset_type": "stock",
    "market_report": "m", "sentiment_report": "s",
    "news_report": "n", "fundamentals_report": "f",
}


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_a_researcher_argues_without_its_scratchpad(factory):
    state = {**_REPORTS, "count": 0, "investment_debate_state": {
        "history": "", "bull_history": "", "bear_history": "", "current_response": "", "count": 0,
    }}
    debate = factory(_thinking_llm())(state)["investment_debate_state"]
    assert "my argument" in debate["history"]
    assert "Maybe Sell" not in debate["history"]
    assert "Maybe Sell" not in debate["current_response"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory", [create_aggressive_debator, create_conservative_debator, create_neutral_debator]
)
def test_a_risk_analyst_argues_without_its_scratchpad(factory):
    state = {**_REPORTS, "trader_investment_plan": "plan", "risk_debate_state": {
        "current_aggressive_response": "", "current_conservative_response": "",
        "current_neutral_response": "", "history": "", "aggressive_history": "",
        "conservative_history": "", "neutral_history": "", "count": 0,
    }}
    debate = factory(_thinking_llm())(state)["risk_debate_state"]
    assert "my argument" in debate["history"]
    assert "Maybe Sell" not in debate["history"]
