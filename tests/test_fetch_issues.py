"""Tracking of degraded/failed data fetches (tradingagents.dataflows.fetch_issues),
and its wiring into interface.route_to_vendor's four "could not serve this
call cleanly" exit points.
"""

import copy
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import fetch_issues, interface
from tradingagents.dataflows.config import set_config


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_config()
    fetch_issues.set_collector(None)
    yield
    _reset_config()
    fetch_issues.set_collector(None)


# --- FetchIssueCollector -------------------------------------------------

@pytest.mark.unit
def test_record_outside_a_cell_has_no_ticker_or_date():
    collector = fetch_issues.FetchIssueCollector()
    collector.record("get_news", "no_data", "nothing found", ("AAPL",))
    assert collector.issues == [{
        "ticker": None, "date": None, "method": "get_news", "reason": "no_data",
        "detail": "nothing found", "args": ["AAPL"],
        "ts": collector.issues[0]["ts"],
    }]


@pytest.mark.unit
def test_record_inside_a_cell_is_tagged():
    collector = fetch_issues.FetchIssueCollector()
    with collector.cell("NVDA", "2026-01-05"):
        collector.record("get_news", "no_data", "nothing found", ("NVDA",))
    assert collector.issues[0]["ticker"] == "NVDA"
    assert collector.issues[0]["date"] == "2026-01-05"


@pytest.mark.unit
def test_the_cell_tag_does_not_leak_past_its_block():
    collector = fetch_issues.FetchIssueCollector()
    with collector.cell("NVDA", "2026-01-05"):
        pass
    collector.record("get_news", "no_data", "x", ())
    assert collector.issues[0]["ticker"] is None


@pytest.mark.unit
def test_nested_cells_restore_the_outer_tag():
    collector = fetch_issues.FetchIssueCollector()
    with collector.cell("NVDA", "2026-01-05"):
        with collector.cell("AAPL", "2026-01-12"):
            collector.record("get_news", "no_data", "inner", ())
        collector.record("get_news", "no_data", "outer", ())
    assert collector.issues[0]["ticker"] == "AAPL"
    assert collector.issues[1]["ticker"] == "NVDA"


# --- module-level record() -------------------------------------------------

@pytest.mark.unit
def test_record_always_logs_even_with_no_active_collector():
    with pytest.MonkeyPatch().context() as mp:
        warn = mock.Mock()
        mp.setattr(fetch_issues.logger, "warning", warn)
        fetch_issues.record("get_news", "no_data", "detail", ("AAPL",))
    warn.assert_called_once()


@pytest.mark.unit
def test_record_feeds_the_active_collector():
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)
    fetch_issues.record("get_news", "no_data", "detail", ("AAPL",))
    assert len(collector.issues) == 1
    assert collector.issues[0]["method"] == "get_news"


@pytest.mark.unit
def test_record_is_a_no_op_for_collection_when_no_collector_is_active():
    fetch_issues.set_collector(None)
    fetch_issues.record("get_news", "no_data", "detail", ())  # must not raise


# --- wiring into route_to_vendor --------------------------------------------

def _no_data_error(symbol="AAPL"):
    from tradingagents.dataflows.errors import NoMarketDataError
    return NoMarketDataError(symbol, symbol, "no rows")


@pytest.mark.unit
def test_no_data_sentinel_is_recorded_as_no_data():
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    def _raise(*a, **k):
        raise _no_data_error()

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": _raise}}, clear=False):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert collector.issues[0]["reason"] == "no_data"
    assert collector.issues[0]["method"] == "get_stock_data"


@pytest.mark.unit
def test_all_vendors_unavailable_is_recorded_as_vendor_unavailable():
    from tradingagents.dataflows.errors import VendorRateLimitError

    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    def _throttled(*a, **k):
        raise VendorRateLimitError("slow down")

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": _throttled}}, clear=False):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert collector.issues[0]["reason"] == "vendor_unavailable"


@pytest.mark.unit
def test_optional_category_degrade_is_recorded_as_vendor_unavailable():
    set_config({"data_vendors": {"macro_data": "fred"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    def _boom(*a, **k):
        raise ValueError("FRED 400: bad series")

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_macro_indicators": {"fred": _boom}}, clear=False):
        interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")

    assert collector.issues[0]["reason"] == "vendor_unavailable"


@pytest.mark.unit
def test_a_clean_success_records_nothing():
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": lambda *a, **k: "OK"}}, clear=False):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert collector.issues == []


@pytest.mark.unit
def test_a_feed_that_never_observed_the_window_is_recorded():
    """News/social feeds report an unobserved window as an ordinary return
    value, so it reaches no router error path — it must still be visible."""
    from datetime import datetime, timezone

    from tradingagents.dataflows.date_window import coverage_gap

    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    gap = coverage_gap(
        [datetime(2026, 5, 20, tzinfo=timezone.utc)],
        "2026-05-01", "2026-05-08", "Yahoo Finance news", "news for AAPL",
    )

    assert gap is not None
    assert collector.issues[0]["reason"] == "no_coverage"
    assert collector.issues[0]["method"] == "Yahoo Finance news"
    assert "2026-05-01..2026-05-08" in collector.issues[0]["detail"]


@pytest.mark.unit
def test_an_observed_window_records_nothing():
    from datetime import datetime, timezone

    from tradingagents.dataflows.date_window import coverage_gap

    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    gap = coverage_gap(
        [datetime(2026, 4, 1, tzinfo=timezone.utc)],
        "2026-05-01", "2026-05-08", "Yahoo Finance news", "news for AAPL",
    )

    assert gap is None
    assert collector.issues == []


@pytest.mark.unit
def test_issue_is_tagged_with_the_cell_the_backtest_set():
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    def _raise(*a, **k):
        raise _no_data_error()

    with collector.cell("NVDA", "2026-03-01"), \
            mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": _raise}}, clear=False):
        interface.route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-01-08")

    assert collector.issues[0]["ticker"] == "NVDA"
    assert collector.issues[0]["date"] == "2026-03-01"


@pytest.mark.unit
def test_an_argument_the_model_invented_is_not_reported_as_an_outage():
    """A bad indicator name comes back as a plain ValueError the model recovers
    from; filed as an error it would outrank the vendor outages in the report."""
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    def _bad_argument(*a, **k):
        raise ValueError("Indicator vwap is not supported. Please choose from: [...]")

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": _bad_argument}}, clear=False), \
            pytest.raises(ValueError):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert collector.issues[0]["reason"] == "unsupported_request"


@pytest.mark.unit
def test_a_broken_vendor_is_still_an_error():
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    collector = fetch_issues.FetchIssueCollector()
    fetch_issues.set_collector(collector)

    def _broken(*a, **k):
        raise ConnectionError("connection reset by peer")

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": _broken}}, clear=False), \
            pytest.raises(ConnectionError):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert collector.issues[0]["reason"] == "error"
