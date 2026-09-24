"""Read-through cache for data-vendor tool calls, so a backtest re-run with a
different model reads the same fetched inputs (news, OHLCV, fundamentals, ...)
instead of hitting live vendors again.

Keyed on the exact call signature (method + args + kwargs), never on which
model/provider is asking, so two backtests on the same (ticker, date) cell with
different LLMs see identical tool output. No TTL: the key already encodes the
requested window, so a cached entry never needs to be judged "stale" the way a
single-file-per-symbol cache (e.g. ``stockstats_utils``) does.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def _cache_dir(config: dict) -> Path:
    return Path(config["data_cache_dir"]) / "fetch_cache"


def cache_key(method: str, args: tuple, kwargs: dict) -> str:
    """Deterministic hash of a call signature; stable across process runs."""
    canonical = json.dumps(
        {"method": method, "args": list(args), "kwargs": kwargs},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _entry_path(method: str, args: tuple, kwargs: dict, config: dict) -> Path:
    return _cache_dir(config) / method / f"{cache_key(method, args, kwargs)}.json"


def read_cached(method: str, args: tuple, kwargs: dict, config: dict) -> str | None:
    """The cached return value for this exact call, or ``None`` on a miss."""
    path = _entry_path(method, args, kwargs, config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["result"]
    except (ValueError, KeyError, OSError):
        return None  # a truncated/corrupt entry is a miss, not a failure


def write_cache(method: str, args: tuple, kwargs: dict, config: dict, result: str) -> None:
    """Store a fetched result. Failing to store must never fail the fetch.

    A full disk, a path the OS rejects, or an argument that will not serialize
    would otherwise abort a sweep that had already got its data.

    An agent can issue the same call twice in one parallel batch, and on Windows
    replacing a file another thread holds open is refused. So the temporary name
    is private to this writer, and an entry that already exists is left alone —
    entries never change once written, so whoever stored it first stored this.
    """
    path = _entry_path(method, args, kwargs, config)
    if path.exists():
        return
    payload = {
        "method": method,
        "args": list(args),
        "kwargs": kwargs,
        "fetched_at": time.time(),
        "result": result,
    }
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not cache %s: %s", method, exc)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def clear_fetch_cache(data_cache_dir: str | Path) -> int:
    """Remove every cached fetch entry. Returns the number of files deleted."""
    root = Path(data_cache_dir) / "fetch_cache"
    if not root.exists():
        return 0
    files = list(root.rglob("*.json"))
    for f in files:
        f.unlink(missing_ok=True)
    return len(files)


def cached_call(
    method: str,
    args: tuple,
    kwargs: dict,
    config: dict,
    fetch_fn: Callable[[], str],
    cacheable: Callable[[object], bool] | None = None,
) -> str:
    """Serve ``method(*args, **kwargs)`` from cache, else fetch and store it.

    A no-op passthrough unless ``config["cache_tool_fetches"]`` is set, so live
    runs keep fetching fresh data. ``cacheable`` vets a fetched result before it
    is stored: a result a caller considers degraded (and an exception, which is
    never stored) must stay out, or one transient vendor failure would be frozen
    in and served to every later run.
    """
    if not config.get("cache_tool_fetches"):
        return fetch_fn()
    cached = read_cached(method, args, kwargs, config)
    if cached is not None:
        return cached
    result = fetch_fn()
    if cacheable is None or cacheable(result):
        write_cache(method, args, kwargs, config, result)
    return result
