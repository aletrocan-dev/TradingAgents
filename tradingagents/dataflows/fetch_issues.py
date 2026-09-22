"""Tracks data fetches that degraded to "no data" / "vendor unavailable" or
raised, so a backtest run can report exactly which (ticker, date, tool) cells
had missing or failed data instead of that fact passing silently into the
agent's prompt as an unremarkable sentinel string.

Uses a ``ContextVar`` rather than an explicit parameter threaded through
``route_to_vendor`` and every tool wrapper, so recording an issue needs no
signature changes anywhere upstream. Outside a tracked ``cell()`` (e.g. a
plain interactive run), issues are still logged, just not collected.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_current_cell: ContextVar[tuple[str | None, str | None]] = ContextVar(
    "_current_cell", default=(None, None)
)
_active_collector: ContextVar[FetchIssueCollector | None] = ContextVar(
    "_active_collector", default=None
)


@dataclass
class FetchIssueCollector:
    """Collects fetch issues recorded while it is the active collector."""

    issues: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def cell(self, ticker: str, date: str) -> Generator[None, None, None]:
        """Tag every issue recorded inside this block with (ticker, date)."""
        token = _current_cell.set((ticker, date))
        try:
            yield
        finally:
            _current_cell.reset(token)

    def record(self, method: str, reason: str, detail: str, args: tuple) -> None:
        ticker, date = _current_cell.get()
        self.issues.append({
            "ticker": ticker,
            "date": date,
            "method": method,
            "reason": reason,
            "detail": detail,
            "args": [str(a) for a in args],
            "ts": time.time(),
        })


def set_collector(collector: FetchIssueCollector | None) -> None:
    """Activate (or clear, with ``None``) the collector ``record`` reports to."""
    _active_collector.set(collector)


def record(method: str, reason: str, detail: str, args: tuple = ()) -> None:
    """Log a fetch issue, and collect it if a collector is active.

    Always logs, so a degraded fetch is visible in real time (console/log)
    even outside a backtest, where no collector is set.
    """
    ticker, date = _current_cell.get()
    where = f" [{ticker} {date}]" if ticker else ""
    logger.warning("Fetch issue%s: %s (%s) — %s", where, method, reason, detail)
    collector = _active_collector.get()
    if collector is not None:
        collector.record(method, reason, detail, args)
