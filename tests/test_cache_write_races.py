"""Parallel tool calls writing the same cache file.

An agent can issue the same call twice in one parallel batch, and indicator
calls all rewrite the day's price file. On Windows, replacing a file another
thread holds open is refused, and a reader that caught a file mid-write got an
empty frame. Neither may fail a fetch that already has its data.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.stockstats_utils as su
from tradingagents.dataflows import fetch_cache


def _refuse(*args):
    raise PermissionError("held open by another thread")


@pytest.mark.unit
def test_an_entry_already_stored_is_left_alone(tmp_path):
    config = {"data_cache_dir": str(tmp_path)}
    fetch_cache.write_cache("get_news", ("BTC",), {}, config, "first")
    fetch_cache.write_cache("get_news", ("BTC",), {}, config, "second")
    assert fetch_cache.read_cached("get_news", ("BTC",), {}, config) == "first"
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.unit
def test_a_failed_store_leaves_no_temporary_file(tmp_path, monkeypatch):
    config = {"data_cache_dir": str(tmp_path)}
    monkeypatch.setattr(fetch_cache.os, "replace", _refuse)
    fetch_cache.write_cache("get_news", ("BTC",), {}, config, "data")  # must not raise
    assert not list(tmp_path.rglob("*.tmp"))
    assert fetch_cache.read_cached("get_news", ("BTC",), {}, config) is None


@pytest.mark.unit
def test_prices_are_replaced_whole_or_not_at_all(tmp_path, monkeypatch):
    path = tmp_path / "BTC-USD.csv"
    path.write_text("Date,Close\n2026-01-07,100\n", encoding="utf-8")
    frame = pd.DataFrame({"Date": ["2026-01-07", "2026-01-08"], "Close": [100, 110]})

    su._write_csv_atomically(frame, str(path))
    assert pd.read_csv(path)["Close"].tolist() == [100, 110]

    monkeypatch.setattr(su.os, "replace", _refuse)
    su._write_csv_atomically(frame.head(1), str(path))  # must not raise
    assert pd.read_csv(path)["Close"].tolist() == [100, 110]  # the old file is intact
    assert not list(tmp_path.glob("*.tmp"))
