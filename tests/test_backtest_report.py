"""The self-contained HTML backtest report (tradingagents.backtest_report)."""


import pytest

from tradingagents.backtest import BacktestResult, BacktestSummary, RatingScore
from tradingagents.backtest_report import write_html_report


def _result(tmp_path, **overrides):
    log_path = tmp_path / "run" / "trading_memory.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    defaults = {"run_id": "20260101_000000", "log_path": log_path, "cells_run": 2, "skipped": 0}
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _summary(**overrides):
    defaults = {
        "resolved": 1, "pending": 0,
        "by_rating": {"Buy": RatingScore(count=1, hit_rate=1.0, mean_value=0.04)},
        "unscored": 0, "holding": "5 trading days",
    }
    defaults.update(overrides)
    return BacktestSummary(**defaults)


_ENTRY = {
    "date": "2026-01-05", "ticker": "NVDA", "rating": "Buy", "pending": False,
    "raw": "+10.0%", "alpha": "+4.0%", "holding": "5d", "resolved": "2026-01-12",
    "decision": "Rating: Buy\n\nbuy it", "reflection": "worked out",
}


@pytest.mark.unit
def test_report_is_written_next_to_the_log(tmp_path):
    result = _result(tmp_path)
    path = write_html_report(result, _summary(), [_ENTRY], {"llm_provider": "openai"})
    assert path == result.log_path.parent / "report.html"
    assert path.exists()


@pytest.mark.unit
def test_report_names_the_run_and_tickers(tmp_path):
    result = _result(tmp_path)
    path = write_html_report(result, _summary(), [_ENTRY], {"llm_provider": "openai"})
    html = path.read_text(encoding="utf-8")
    assert "20260101_000000" in html
    assert "NVDA" in html


@pytest.mark.unit
def test_report_handles_no_entries_without_crashing(tmp_path):
    result = _result(tmp_path)
    summary = BacktestSummary(resolved=0, pending=0, by_rating={}, unscored=0, holding="")
    path = write_html_report(result, summary, [], {"llm_provider": "openai"})
    assert path.exists()
    assert "Nessun" in path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_report_handles_all_pending_entries_without_crashing(tmp_path):
    result = _result(tmp_path)
    pending_entry = {**_ENTRY, "pending": True, "alpha": None, "raw": None}
    summary = BacktestSummary(resolved=0, pending=1, by_rating={}, unscored=0, holding="")
    path = write_html_report(result, summary, [pending_entry], {"llm_provider": "openai"})
    assert path.exists()


@pytest.mark.unit
def test_fetch_issues_appear_in_the_report(tmp_path):
    result = _result(tmp_path, fetch_issues=[{
        "ticker": "NVDA", "date": "2026-01-05", "method": "get_global_news",
        "reason": "no_data", "detail": "nothing in window", "args": [], "ts": 0.0,
    }])
    path = write_html_report(result, _summary(), [_ENTRY], {"llm_provider": "openai"})
    html = path.read_text(encoding="utf-8")
    assert "get_global_news" in html
    assert "no_data" in html


@pytest.mark.unit
def test_no_fetch_issues_says_so_explicitly(tmp_path):
    result = _result(tmp_path)
    path = write_html_report(result, _summary(), [_ENTRY], {"llm_provider": "openai"})
    html = path.read_text(encoding="utf-8")
    assert "Nessun problema di fetch" in html


@pytest.mark.unit
def test_failures_appear_in_the_report(tmp_path):
    result = _result(tmp_path, failures=[("NVDA", "2026-01-05", "vendor exploded")])
    path = write_html_report(result, _summary(), [_ENTRY], {"llm_provider": "openai"})
    html = path.read_text(encoding="utf-8")
    assert "vendor exploded" in html


@pytest.mark.unit
def test_report_is_a_single_file_with_no_external_references(tmp_path):
    """Must open and read correctly offline: no CDN scripts/stylesheets."""
    result = _result(tmp_path)
    path = write_html_report(result, _summary(), [_ENTRY], {"llm_provider": "openai"})
    html = path.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html


_REVIEW_ENTRY = {
    "date": "2026-01-06", "ticker": "NVDA", "rating": "REVIEW", "pending": False,
    "raw": "+6.0%", "alpha": "+4.9%", "holding": "5d", "resolved": "2026-01-13",
    "decision": "no readable rating", "reflection": "",
}


@pytest.mark.unit
def test_a_review_cell_is_not_plotted_as_resolved(tmp_path):
    """It settled, but its rating is unreadable, so the summary counts it as
    unscored — plotting it would contradict the cards above it."""
    result = _result(tmp_path)
    summary = BacktestSummary(resolved=0, pending=0, by_rating={}, unscored=1, holding="")
    path = write_html_report(result, summary, [_REVIEW_ENTRY], {"llm_provider": "openai"})
    html = path.read_text(encoding="utf-8")
    assert "REVIEW · alpha" not in html
    assert "Nessuna cella risolta" in html


@pytest.mark.unit
def test_a_scored_cell_is_still_plotted(tmp_path):
    path = write_html_report(_result(tmp_path), _summary(), [_ENTRY], {"llm_provider": "openai"})
    assert "NVDA 2026-01-05" in path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_repeated_issue_is_one_row_with_a_count(tmp_path):
    """One missing API key hit once per macro series is one problem, not six."""
    issue = {
        "ticker": "NVDA", "date": "2026-01-05", "method": "get_macro_indicators",
        "reason": "vendor_unavailable", "detail": "FRED_API_KEY is not set",
        "args": [], "ts": 0.0,
    }
    result = _result(tmp_path, fetch_issues=[dict(issue) for _ in range(6)])
    html = write_html_report(
        result, _summary(), [_ENTRY], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")

    assert html.count("FRED_API_KEY is not set") == 1
    assert "×6" in html


@pytest.mark.unit
def test_distinct_problems_on_the_same_tool_stay_separate(tmp_path):
    result = _result(tmp_path, fetch_issues=[
        {"ticker": "NVDA", "date": "2026-01-05", "method": "get_news",
         "reason": "no_coverage", "detail": "window A", "args": [], "ts": 0.0},
        {"ticker": "NVDA", "date": "2026-01-06", "method": "get_news",
         "reason": "no_coverage", "detail": "window B", "args": [], "ts": 0.0},
    ])
    html = write_html_report(
        result, _summary(), [_ENTRY], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")
    assert "window A" in html and "window B" in html


@pytest.mark.unit
def test_an_invented_argument_is_not_dressed_as_an_outage(tmp_path):
    """The model recovers from it on the next call; painting it like a broken
    vendor would bury the real failures."""
    from tradingagents.backtest_report import _reason_color

    assert _reason_color("unsupported_request") != _reason_color("error")
    result = _result(tmp_path, fetch_issues=[{
        "ticker": "NVDA", "date": "2026-01-05", "method": "get_indicators",
        "reason": "unsupported_request", "detail": "Indicator vwap is not supported",
        "args": [], "ts": 0.0,
    }])
    html = write_html_report(
        result, _summary(), [_ENTRY], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")
    assert _reason_color("unsupported_request") in html


@pytest.mark.unit
def test_the_holding_note_reads_in_one_language(tmp_path):
    html = write_html_report(
        _result(tmp_path), _summary(), [_ENTRY], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")
    assert "5 giorni di trading" in html
    assert "trading days" not in html


@pytest.mark.unit
def test_the_holding_note_falls_back_without_resolved_cells(tmp_path):
    summary = BacktestSummary(resolved=0, pending=1, by_rating={}, unscored=0, holding="")
    pending_entry = {**_ENTRY, "pending": True, "alpha": None}
    html = write_html_report(
        _result(tmp_path), summary, [pending_entry], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")
    assert "la finestra configurata" in html


@pytest.mark.unit
def test_the_chart_marks_what_each_decision_did(tmp_path):
    """A reader looks for where the system acted and what it called; inferring
    it from a bend in the line is guesswork."""
    from tradingagents.strategy_curve import StrategyCurve

    curve = StrategyCurve(
        ticker="BTC-USD", dates=["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        strategy=[1.0, 1.05, 1.02, 1.02], buy_hold=[1.0, 1.05, 1.02, 0.99],
        positions=[1.0, 1.0, 0.0], markers=[(0, "Buy"), (2, "Sell"), (3, "REVIEW")],
        rule={"Buy": 1.0, "Sell": 0.0},
    )
    html = write_html_report(
        _result(tmp_path, curves=[curve]), _summary(), [_ENTRY], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")

    assert "2026-01-01: Buy" in html      # tooltip sul marcatore
    assert "2026-01-03: Sell" in html
    assert "var(--dir-up)" in html and "var(--dir-down)" in html
    assert "fill='none'" in html          # REVIEW = marcatore vuoto
    assert "compra" in html and "vendi" in html and "mantiene" in html


@pytest.mark.unit
def test_direction_is_carried_by_shape_and_not_by_colour_alone():
    """Green and red measure DeltaE 7.2 apart under protanopia, which the palette
    validator only allows with a second channel — here, the marker's shape."""
    from tradingagents.backtest_report import _marker_shape

    up = _marker_shape("Buy", 10, 10, "var(--dir-up)")
    down = _marker_shape("Sell", 10, 10, "var(--dir-down)")
    flat = _marker_shape("Hold", 10, 10, "var(--dir-flat)")
    review = _marker_shape("REVIEW", 10, 10, "var(--dir-unknown)")

    assert "polygon" in up and "polygon" in down
    assert up.split("points=")[1] != down.split("points=")[1]   # su vs giu'
    assert "circle" in flat and "fill='none'" not in flat
    assert "fill='none'" in review                              # vuoto: niente deciso


@pytest.mark.unit
def test_the_chart_states_how_long_the_backtest_ran(tmp_path):
    from tradingagents.strategy_curve import StrategyCurve

    curve = StrategyCurve(
        ticker="BTC-USD", dates=["2026-01-01", "2026-01-02", "2026-01-03"],
        strategy=[1.0, 1.05, 1.02], buy_hold=[1.0, 1.05, 1.02],
        positions=[1.0, 1.0], markers=[(0, "Buy")],
    )
    html = write_html_report(
        _result(tmp_path, curves=[curve]), _summary(), [_ENTRY], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")

    assert "3 giorni di prezzo" in html
    assert "2026-01-01" in html and "2026-01-03" in html
    assert "1 decisioni" in html


@pytest.mark.unit
def test_every_decision_is_listed_with_its_outcome(tmp_path):
    """The report should say what happened, not only summarise it."""
    pending = {**_ENTRY, "date": "2026-01-12", "pending": True, "raw": None, "alpha": None}
    review = {**_ENTRY, "date": "2026-01-19", "rating": "REVIEW"}
    html = write_html_report(
        _result(tmp_path), _summary(), [_ENTRY, pending, review], {"llm_provider": "openai"}
    ).read_text(encoding="utf-8")

    for date in ("2026-01-05", "2026-01-12", "2026-01-19"):
        assert date in html
    assert "+4.0%" in html            # l'alpha della cella risolta
    assert "in attesa" in html        # la pending si distingue
    assert "non valutabile" in html   # e la REVIEW pure
