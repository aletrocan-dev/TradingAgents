from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.dataflows import fetch_cache
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import as_of, canonical_span
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.interface import _is_cacheable, no_data_result
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int, "number of recent trading rows to include for sanity-checking"
    ] = 30,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """Deterministic verification snapshot for exact market-data claims.

    Returns the latest OHLCV row on or before curr_date, common technical
    indicators, and recent closes. Call this before making exact claims about
    price levels, Bollinger bands, RSI, MACD, moving averages, support /
    resistance, or historical comparisons, and treat it as the source of truth.
    """
    date = as_of(curr_date, trade_date)
    look_back_days = canonical_span(look_back_days, 30)
    args = (symbol, date, look_back_days)

    def snapshot() -> str:
        # Not vendor-routed, so it answers an unknown symbol the way the router
        # does: raised, it ended the whole cell when a model mistyped a ticker.
        try:
            return build_verified_market_snapshot(symbol, date, look_back_days)
        except NoMarketDataError as exc:
            return no_data_result("get_verified_market_snapshot", exc, args)

    # Cached like the routed tools when the run asks for it: this is the market
    # analyst's ground truth, and the OHLCV download under it is refreshed daily
    # (split/dividend re-adjustment moves historical closes), so without this a
    # re-run would compare models against different numbers. A no-data verdict
    # stays out of the cache, as it does for the routed tools.
    return fetch_cache.cached_call(
        "get_verified_market_snapshot", args, {}, get_config(), snapshot,
        cacheable=_is_cacheable,
    )
