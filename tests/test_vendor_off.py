"""A data category with no vendor is off, and its tool is never offered.

Polymarket is unreachable from some networks (an ISP that resolves its host to
an address that is not it). Leaving the tool bound costs a tool call and its
round trip on every run, to rediscover a failure that is not going to change.
"""

import contextlib
import copy

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import vendor_configured


@pytest.fixture(autouse=True)
def _isolate():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.mark.unit
@pytest.mark.parametrize("value, expected", [
    ("polymarket", True),
    ("", False),
    (None, False),
])
def test_a_category_is_off_when_it_has_no_vendor(value, expected):
    set_config({"data_vendors": {"prediction_markets": value}})
    assert vendor_configured("prediction_markets") is expected


@pytest.mark.unit
def test_the_news_analyst_drops_the_tool_when_the_category_is_off(mock_llm_client):
    from tradingagents.agents.analysts import news_analyst

    bound = {}

    class _LLM:
        def bind_tools(self, tools):
            bound["names"] = [t.name for t in tools]
            return self

    set_config({"data_vendors": {"prediction_markets": ""}})
    state = {"trade_date": "2026-09-22", "company_of_interest": "BTC-USD",
             "asset_type": "crypto", "messages": []}
    # the fake LLM cannot answer; the binding is what matters
    with contextlib.suppress(Exception):
        news_analyst.create_news_analyst(_LLM())(state)

    assert "get_prediction_markets" not in bound["names"]
    assert "get_macro_indicators" in bound["names"]


@pytest.mark.unit
def test_the_news_analyst_offers_the_tool_when_a_vendor_is_configured(mock_llm_client):
    from tradingagents.agents.analysts import news_analyst

    bound = {}

    class _LLM:
        def bind_tools(self, tools):
            bound["names"] = [t.name for t in tools]
            return self

    set_config({"data_vendors": {"prediction_markets": "polymarket"}})
    state = {"trade_date": "2026-09-22", "company_of_interest": "BTC-USD",
             "asset_type": "crypto", "messages": []}
    with contextlib.suppress(Exception):
        news_analyst.create_news_analyst(_LLM())(state)

    assert "get_prediction_markets" in bound["names"]


@pytest.mark.unit
@pytest.mark.parametrize("vendor, mentioned", [("polymarket", True), ("", False)])
def test_the_prompt_only_advertises_tools_the_model_has(vendor, mentioned):
    """Naming a tool the model was not given invites it to try and fail."""
    import langchain_core.prompts as lcp

    from tradingagents.agents.analysts import news_analyst

    class _LLM:
        def bind_tools(self, tools):
            return self

    captured: dict = {}
    original = lcp.ChatPromptTemplate.partial

    def _spy(self, **kwargs):
        captured.update(kwargs)
        return original(self, **kwargs)

    set_config({"data_vendors": {"prediction_markets": vendor}})
    lcp.ChatPromptTemplate.partial = _spy
    try:
        news_analyst.create_news_analyst(_LLM())(
            {"trade_date": "2026-09-22", "company_of_interest": "BTC-USD",
             "asset_type": "crypto", "messages": []}
        )
    except Exception:
        pass  # the fake LLM cannot answer; the prompt is what matters
    finally:
        lcp.ChatPromptTemplate.partial = original

    assert ("get_prediction_markets" in captured["system_message"]) is mentioned
    assert "get_macro_indicators" in captured["system_message"]
