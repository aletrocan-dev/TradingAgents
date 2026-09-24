"""Alpha measures the benchmark over the same days the asset was held.

Counting bars on both series measured them over different spans whenever the
two keep different calendars (crypto trades every day, an equity index five a
week), and comparing timestamps lost the index bar for the same day because
each exchange stamps its daily bar at its own midnight. A weekend holding then
had no alpha at all, instead of a zero index move.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog, _pct
from tradingagents.graph import trading_graph as tg
from tradingagents.graph.reflection import Reflector
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _bars(closes: dict[str, float], tz: str) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(list(closes))).tz_localize(tz)
    return pd.DataFrame({"Close": list(closes.values())}, index=index)


# SPY prints Monday to Friday, stamped at New York midnight.
_SPY = _bars({
    "2026-01-02": 500.0, "2026-01-05": 501.0, "2026-01-06": 502.0, "2026-01-07": 500.0,
    "2026-01-08": 505.0, "2026-01-09": 510.0, "2026-01-12": 520.0,
}, "America/New_York")


def _yahoo(monkeypatch, frames: dict[str, pd.DataFrame]):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, *args, **kwargs):
            if self.symbol not in frames:
                raise ConnectionError("unreachable")
            return frames[self.symbol]

    monkeypatch.setattr(tg.yf, "Ticker", FakeTicker)


def _spy_return(first: date, last: date):
    return TradingAgentsGraph._benchmark_return(
        "SPY", first, last, datetime.combine(first, datetime.min.time()), "2026-01-20",
    )


@pytest.mark.unit
def test_held_friday_to_saturday_the_index_did_not_move(monkeypatch):
    """The market was shut: a zero return, not a missing one."""
    _yahoo(monkeypatch, {"SPY": _SPY})
    assert _spy_return(date(2026, 1, 9), date(2026, 1, 10)) == 0.0


@pytest.mark.unit
def test_held_over_a_weekend_the_index_is_priced_on_its_own_calendar(monkeypatch):
    _yahoo(monkeypatch, {"SPY": _SPY})
    ret = _spy_return(date(2026, 1, 10), date(2026, 1, 12))
    assert ret == pytest.approx(520.0 / 510.0 - 1)  # Friday's close to Monday's


@pytest.mark.unit
def test_an_unpriceable_benchmark_is_none_rather_than_an_error(monkeypatch):
    _yahoo(monkeypatch, {"SPY": pd.DataFrame()})
    assert _spy_return(date(2026, 1, 9), date(2026, 1, 10)) is None
    _yahoo(monkeypatch, {})
    assert _spy_return(date(2026, 1, 9), date(2026, 1, 10)) is None


@pytest.mark.unit
def test_a_crypto_day_is_matched_to_the_index_day_despite_the_time_zones(monkeypatch):
    """BTC's bar is stamped at UTC midnight and SPY's hours later: comparing
    instants rather than calendar days lost the index bar for the same day."""
    btc = _bars({"2026-01-07": 100.0, "2026-01-08": 110.0}, "UTC")
    _yahoo(monkeypatch, {"BTC-USD": btc, "SPY": _SPY})
    raw, alpha, days, resolved = TradingAgentsGraph._fetch_returns(
        None, "BTC-USD", "2026-01-07", holding_days=1, benchmark="SPY",
    )
    assert raw == pytest.approx(0.10)
    assert alpha == pytest.approx(0.10 - (505.0 / 500.0 - 1))
    assert (days, resolved) == (1, "2026-01-08")


@pytest.mark.unit
def test_the_asset_settles_even_when_the_benchmark_cannot_be_priced(monkeypatch):
    btc = _bars({"2026-01-07": 100.0, "2026-01-08": 110.0}, "UTC")
    _yahoo(monkeypatch, {"BTC-USD": btc})
    raw, alpha, days, resolved = TradingAgentsGraph._fetch_returns(
        None, "BTC-USD", "2026-01-07", holding_days=1, benchmark="SPY",
    )
    assert raw == pytest.approx(0.10)
    assert alpha is None
    assert resolved == "2026-01-08"


@pytest.mark.unit
@pytest.mark.parametrize("tz", [None, "UTC", "America/New_York"])
def test_a_bar_still_trading_today_does_not_settle_a_cell(monkeypatch, tz):
    """Settling on today's bar would score a price that is still moving."""
    today = pd.Timestamp.now(tz=tz).normalize()
    index = pd.DatetimeIndex([today - pd.Timedelta(days=1), today])
    btc = pd.DataFrame({"Close": [100.0, 110.0]}, index=index)
    _yahoo(monkeypatch, {"BTC-USD": btc, "SPY": _SPY})
    trade_date = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert TradingAgentsGraph._fetch_returns(
        None, "BTC-USD", trade_date, holding_days=1, benchmark="SPY",
    ) == (None, None, None, None)


@pytest.mark.unit
def test_a_missing_alpha_is_logged_as_not_available(tmp_path):
    assert _pct(None) == "n/a"
    assert _pct(0.1234) == "+12.3%"
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "log.md")})
    log.store_decision("BTC-USD", "2026-01-07", "Rating: Buy\n\nbuy")
    log.batch_update_with_outcomes([{
        "ticker": "BTC-USD", "trade_date": "2026-01-07", "raw_return": 0.1,
        "alpha_return": None, "holding_days": 1, "reflection": "", "resolution_date": "2026-01-08",
    }])
    (entry,) = log.load_entries()
    assert (entry["raw"], entry["alpha"]) == ("+10.0%", "n/a")


@pytest.mark.unit
def test_a_reflection_without_alpha_leaves_the_alpha_line_out():
    llm = MagicMock()
    llm.invoke.return_value.content = "ok"
    Reflector(llm).reflect_on_final_decision("Rating: Buy", raw_return=0.1, alpha_return=None)
    human = next(content for role, content in llm.invoke.call_args[0][0] if role == "human")
    assert "+10.0%" in human
    assert "Alpha" not in human
