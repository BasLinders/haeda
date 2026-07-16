"""Smoke test for pages/forecasting.py: the page should load and run the
one-click demo-data pipeline (fit + chart + metrics) without raising.

Requires `first-order-engine` (and its `prophet`/`cmdstan` backend) to be
installed, per requirements.txt. Skipped if that's not available, since
compiling cmdstan needs a C++ toolchain that not every dev machine has.
"""
from pathlib import Path

import pytest

pytest.importorskip("foe", reason="first-order-engine not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "forecasting.py")


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
    subheaders = [h.value for h in at.subheader]
    assert "Forecast: conversions" in subheaders
    assert "Forecast: revenue" in subheaders
