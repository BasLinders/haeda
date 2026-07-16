"""Smoke test for pages/rfm_analysis.py: the page should load, generate its
built-in mock dataset, and run the full RFM + predictive analysis (BG/NBD and
Gamma-Gamma) without raising."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "rfm_analysis.py")


def test_loads_without_data():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    assert not at.exception
    assert "RFM Analysis" in at.title[0].value


def test_full_analysis_on_mock_data():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    load_mock_button = next(b for b in at.sidebar.button if b.label == "Load Mock Data")
    load_mock_button.click().run(timeout=60)
    assert not at.exception

    run_button = next(b for b in at.button if b.label == "Run Full RFM & Predictive Analysis")
    run_button.click().run(timeout=120)

    assert not at.exception
    subheaders = [h.value for h in at.subheader]
    assert "Customer Insights Table" in subheaders
    assert any("Predictive Analytics" in h for h in subheaders)
    assert len(at.dataframe) > 0
