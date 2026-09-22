"""Read-through cache for data-vendor tool calls (tradingagents.dataflows.fetch_cache).

Two levels: the module's own read/write/clear primitives, and its wiring into
``interface.route_to_vendor`` behind ``config["cache_tool_fetches"]``.
"""

import copy
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import fetch_cache, interface
from tradingagents.dataflows.config import set_config


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_config()
    yield
    _reset_config()


def _config(tmp_path):
    return {"data_cache_dir": str(tmp_path), "cache_tool_fetches": True}


# --- module primitives -------------------------------------------------

@pytest.mark.unit
def test_a_miss_returns_none(tmp_path):
    assert fetch_cache.read_cached("get_news", ("AAPL", "2026-01-01", "2026-01-08"), {}, _config(tmp_path)) is None


@pytest.mark.unit
def test_write_then_read_round_trips(tmp_path):
    config = _config(tmp_path)
    args = ("AAPL", "2026-01-01", "2026-01-08")
    fetch_cache.write_cache("get_news", args, {}, config, "## AAPL news\n...")
    assert fetch_cache.read_cached("get_news", args, {}, config) == "## AAPL news\n..."


@pytest.mark.unit
def test_different_args_are_different_entries(tmp_path):
    config = _config(tmp_path)
    fetch_cache.write_cache("get_news", ("AAPL", "2026-01-01", "2026-01-08"), {}, config, "AAPL result")
    assert fetch_cache.read_cached("get_news", ("NVDA", "2026-01-01", "2026-01-08"), {}, config) is None


@pytest.mark.unit
def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    config = _config(tmp_path)
    args = ("AAPL", "2026-01-01", "2026-01-08")
    fetch_cache.write_cache("get_news", args, {}, config, "ok")
    path = tmp_path / "fetch_cache" / "get_news" / f"{fetch_cache.cache_key('get_news', args, {})}.json"
    path.write_text("not json{{", encoding="utf-8")
    assert fetch_cache.read_cached("get_news", args, {}, config) is None


@pytest.mark.unit
def test_cached_call_only_invokes_fetch_fn_once(tmp_path):
    config = _config(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return "fresh result"

    first = fetch_cache.cached_call("get_news", ("AAPL",), {}, config, fetch)
    second = fetch_cache.cached_call("get_news", ("AAPL",), {}, config, fetch)

    assert first == second == "fresh result"
    assert len(calls) == 1


@pytest.mark.unit
def test_cached_call_is_a_passthrough_when_caching_is_off(tmp_path):
    config = {"data_cache_dir": str(tmp_path)}  # cache_tool_fetches unset
    calls = []

    def fetch():
        calls.append(1)
        return "fresh"

    fetch_cache.cached_call("get_news", ("AAPL",), {}, config, fetch)
    fetch_cache.cached_call("get_news", ("AAPL",), {}, config, fetch)

    assert len(calls) == 2
    assert not (tmp_path / "fetch_cache").exists()


@pytest.mark.unit
def test_cached_call_does_not_cache_a_raised_exception(tmp_path):
    config = _config(tmp_path)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return "recovered"

    with pytest.raises(RuntimeError):
        fetch_cache.cached_call("get_news", ("AAPL",), {}, config, flaky)

    # A failed call must not be frozen in: the next call retries live.
    assert fetch_cache.cached_call("get_news", ("AAPL",), {}, config, flaky) == "recovered"
    assert len(attempts) == 2


@pytest.mark.unit
def test_a_result_the_caller_rejects_is_not_cached(tmp_path):
    config = _config(tmp_path)
    attempts = []

    def fetch():
        attempts.append(1)
        return "DEGRADED" if len(attempts) == 1 else "good data"

    first = fetch_cache.cached_call(
        "get_news", ("AAPL",), {}, config, fetch, cacheable=lambda r: r != "DEGRADED"
    )
    second = fetch_cache.cached_call(
        "get_news", ("AAPL",), {}, config, fetch, cacheable=lambda r: r != "DEGRADED"
    )

    assert first == "DEGRADED"      # served to the agent...
    assert second == "good data"    # ...but never frozen into the cache
    assert len(attempts) == 2


@pytest.mark.unit
def test_a_cache_write_failure_does_not_break_the_fetch(tmp_path, monkeypatch):
    """The data is already in hand; a full disk must not abort a long sweep."""
    config = _config(tmp_path)

    def _boom(*a, **k):
        raise OSError("no space left on device")

    monkeypatch.setattr(fetch_cache.Path, "write_text", _boom)
    assert fetch_cache.cached_call("get_news", ("AAPL",), {}, config, lambda: "data") == "data"


@pytest.mark.unit
def test_clear_fetch_cache_removes_entries_and_counts_them(tmp_path):
    config = _config(tmp_path)
    fetch_cache.write_cache("get_news", ("AAPL",), {}, config, "a")
    fetch_cache.write_cache("get_stock_data", ("NVDA",), {}, config, "b")

    removed = fetch_cache.clear_fetch_cache(tmp_path)

    assert removed == 2
    assert fetch_cache.read_cached("get_news", ("AAPL",), {}, config) is None


@pytest.mark.unit
def test_clear_fetch_cache_on_a_missing_dir_is_a_no_op(tmp_path):
    assert fetch_cache.clear_fetch_cache(tmp_path / "never_created") == 0


# --- wiring into route_to_vendor ----------------------------------------

def _returns(value):
    def impl(*a, **k):
        return value
    return impl


@pytest.mark.unit
def test_route_to_vendor_does_not_cache_by_default(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "yfinance"}})
    vendor = mock.Mock(side_effect=_returns("DATA"))
    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": vendor}}, clear=False):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")
    assert vendor.call_count == 2  # cache disabled: every call hits the vendor


@pytest.mark.unit
def test_route_to_vendor_caches_when_enabled(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "data_vendors": {"core_stock_apis": "yfinance"},
        "cache_tool_fetches": True,
    })
    vendor = mock.Mock(side_effect=_returns("DATA"))
    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": vendor}}, clear=False):
        first = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")
        second = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")
    assert first == second == "DATA"
    assert vendor.call_count == 1  # the second call was served from cache


@pytest.mark.unit
def test_a_no_data_verdict_is_not_frozen_into_the_cache(tmp_path):
    """A vendor that had nothing this time may have it next time; caching the
    verdict would serve the degraded run's input to every later run."""
    from tradingagents.dataflows.errors import NoMarketDataError

    set_config({"data_cache_dir": str(tmp_path), "cache_tool_fetches": True,
                "data_vendors": {"core_stock_apis": "yfinance"}})
    attempts = []

    def flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise NoMarketDataError("AAPL", "AAPL", "no rows")
        return "REAL DATA"

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": flaky}}, clear=False):
        first = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")
        second = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert "NO_DATA_AVAILABLE" in first
    assert second == "REAL DATA"


@pytest.mark.unit
def test_a_throttled_verdict_is_not_frozen_into_the_cache(tmp_path):
    from tradingagents.dataflows.errors import VendorRateLimitError

    set_config({"data_cache_dir": str(tmp_path), "cache_tool_fetches": True,
                "data_vendors": {"core_stock_apis": "yfinance"}})
    attempts = []

    def throttled_once(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise VendorRateLimitError("slow down")
        return "REAL DATA"

    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": throttled_once}}, clear=False):
        first = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")
        second = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert "DATA_UNAVAILABLE" in first
    assert second == "REAL DATA"


@pytest.mark.unit
def test_the_verified_market_snapshot_is_cached_too(tmp_path):
    """The market analyst's ground truth sits outside route_to_vendor, and the
    OHLCV download under it is refreshed daily, so it needs pinning as well."""
    import tradingagents.agents.utils.market_data_validation_tools as tools

    set_config({"data_cache_dir": str(tmp_path), "cache_tool_fetches": True})
    builder = mock.Mock(return_value="SNAPSHOT")
    with mock.patch.object(tools, "build_verified_market_snapshot", builder):
        first = tools.get_verified_market_snapshot.func("AAPL", "2026-01-05", 30, "2026-01-05")
        second = tools.get_verified_market_snapshot.func("AAPL", "2026-01-05", 30, "2026-01-05")

    assert first == second == "SNAPSHOT"
    assert builder.call_count == 1


@pytest.mark.unit
def test_route_to_vendor_cache_is_keyed_on_the_call_not_the_caller(tmp_path):
    """Two different config setups (standing in for two different models/runs)
    hitting the same cache dir with the same call must see the same input."""
    shared_cache_dir = str(tmp_path)
    vendor = mock.Mock(side_effect=_returns("SAME_INPUT"))
    with mock.patch.dict(interface.VENDOR_METHODS, {"get_stock_data": {"yfinance": vendor}}, clear=False):
        set_config({"data_cache_dir": shared_cache_dir, "cache_tool_fetches": True,
                    "data_vendors": {"core_stock_apis": "yfinance"}, "deep_think_llm": "model-a"})
        first = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

        _reset_config()
        set_config({"data_cache_dir": shared_cache_dir, "cache_tool_fetches": True,
                    "data_vendors": {"core_stock_apis": "yfinance"}, "deep_think_llm": "model-b"})
        second = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-08")

    assert first == second == "SAME_INPUT"
    assert vendor.call_count == 1
