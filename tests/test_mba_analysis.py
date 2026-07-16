"""Smoke test for pages/mba_analysis.py: the page should load and run a full
analysis on a small transactions CSV without raising."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "mba_analysis.py")

SAMPLE_CSV = b"""transaction_id,item,category
1,bread,bakery
1,milk,dairy
2,bread,bakery
2,butter,dairy
3,bread,bakery
3,milk,dairy
3,butter,dairy
4,milk,dairy
5,bread,bakery
5,milk,dairy
"""


def test_loads_without_upload():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "🛒 Market Basket Analysis"


def test_full_analysis_on_sample_data():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    at.get("file_uploader")[0].upload("transactions.csv", SAMPLE_CSV, "text/csv")
    at.run(timeout=30)
    assert not at.exception

    analyze_button = next(b for b in at.sidebar.button if b.label == "Analyze")
    analyze_button.click().run(timeout=30)

    assert not at.exception
    headers = [h.value for h in at.header]
    assert any("Analysis Results" in h for h in headers)
    assert any("Explanation of the Strongest Rule" in h for h in headers)
