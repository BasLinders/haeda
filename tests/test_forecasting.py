"""Smoke test for pages/forecasting.py: the page should load and run the
one-click demo-data pipeline (fit + chart + metrics) without raising.

Requires `first-order-engine` (and its `prophet`/`cmdstan` backend) to be
installed, per requirements.txt. Skipped if that's not available, since
compiling cmdstan needs a C++ toolchain that not every dev machine has.
"""
import datetime
import random
from pathlib import Path

import pytest

pytest.importorskip("foe", reason="first-order-engine not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "forecasting.py")


def _synthetic_daily_csv(n_days: int = 500, seed: int = 1) -> bytes:
    """A deterministic daily conversions/revenue CSV, long enough to fit
    weekly/monthly granularity and a year-over-year comparison."""
    rng = random.Random(seed)
    lines = ["date,conversions,revenue"]
    d = datetime.date(2023, 1, 1)
    for i in range(n_days):
        conv = 20 + i * 0.05 + rng.uniform(-3, 3)
        rev = conv * 18 + rng.uniform(-20, 20)
        lines.append(f"{d.isoformat()},{conv:.1f},{rev:.2f}")
        d += datetime.timedelta(days=1)
    return "\n".join(lines).encode("utf-8")


def _upload_and_map(at: AppTest, csv_bytes: bytes) -> AppTest:
    """Upload the synthetic CSV and run once so column-mapping widgets exist."""
    at.get("file_uploader")[0].upload("data.csv", csv_bytes, "text/csv")
    at.run(timeout=30)
    return at


def test_loads_without_upload():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    assert not at.exception
    assert "Demand Forecasting" in at.title[0].value


def test_demo_data_runs_full_pipeline():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    demo_button = next(b for b in at.sidebar.button if "Generate demo data" in b.label)
    demo_button.click().run(timeout=120)

    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    assert "Forecast: Conversions" in tab_labels
    assert "Forecast: Revenue" in tab_labels


def test_upload_weekly_granularity_matches_requested_horizon():
    """Regression test: weekly/monthly granularity used to miscount a
    trailing partial historical bucket as an extra forecast period (e.g.
    "next 9 weeks" for an 8-week request) because the history/future cutoff
    was computed from raw daily dates instead of the resampled series."""
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)
    _upload_and_map(at, _synthetic_daily_csv())

    granularity = next(r for r in at.radio if r.label == "Granularity")
    granularity.set_value("weekly").run(timeout=30)
    periods = next(n for n in at.number_input if "ahead to forecast" in n.label)
    periods.set_value(8).run(timeout=30)

    run_button = next(b for b in at.button if b.label == "Run forecast")
    run_button.click().run(timeout=90)

    assert not at.exception
    total_metric = next(m for m in at.metric if m.label.startswith("Total forecast"))
    assert total_metric.label == "Total forecast, next 8 weeks"
    accuracy_metric = next(m for m in at.metric if m.label == "Accuracy checked over")
    assert accuracy_metric.value == "8 weeks"


def test_upload_single_target_revenue_only():
    """Forecasting revenue without conversions should produce exactly one
    tab (plus the report tab), and the Full report tab shouldn't choke on
    only having one target to combine."""
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)
    _upload_and_map(at, _synthetic_daily_csv(n_days=120))

    forecast_conversions = next(c for c in at.checkbox if c.label == "Forecast conversions")
    forecast_conversions.set_value(False).run(timeout=30)
    forecast_revenue = next(c for c in at.checkbox if c.label == "Forecast revenue")
    forecast_revenue.set_value(True).run(timeout=30)

    run_button = next(b for b in at.button if b.label == "Run forecast")
    run_button.click().run(timeout=90)

    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["Forecast: Revenue", "📄 Full report"]

    # The Full report tab's combined CSV/markdown shouldn't error with a
    # single target.
    download_labels = [b.label for b in at.get("download_button")]
    assert "Download combined forecast as CSV" in download_labels
    assert "Download full report (Markdown)" in download_labels


def test_settings_change_after_run_shows_staleness_warning():
    """Regression test: changing a fit-affecting setting (e.g. horizon)
    after a forecast has already run used to leave the old chart/table
    displayed with no indication it no longer matches the current settings."""
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)
    _upload_and_map(at, _synthetic_daily_csv())

    run_button = next(b for b in at.button if b.label == "Run forecast")
    run_button.click().run(timeout=90)
    assert not at.exception
    assert not [w for w in at.warning if "settings have changed" in w.value]

    periods = next(n for n in at.number_input if "ahead to forecast" in n.label)
    periods.set_value(15).run(timeout=30)

    stale_warnings = [w for w in at.warning if "settings have changed" in w.value]
    assert stale_warnings, "expected a staleness warning after changing the horizon"
    # The old result (30 days) should still be the one on screen, untouched.
    total_metric = next(m for m in at.metric if m.label.startswith("Total forecast"))
    assert total_metric.label == "Total forecast, next 30 days"
