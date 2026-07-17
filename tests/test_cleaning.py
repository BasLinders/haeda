"""Unit tests for forecast_utils/cleaning.py — pure functions, no Streamlit
or first-order-engine required, so these run fast and everywhere.
"""
import io

import pandas as pd
import pytest

from forecast_utils.cleaning import (
    best_effort_parse_dates,
    clean_numeric_string,
    clean_uploaded_dataframe,
    drop_empty_rows_and_columns,
    finalize_for_fit,
    normalize_columns,
    read_csv_robust,
    suggest_date_column,
    suggest_numeric_columns,
)


class _FakeUpload:
    """Minimal stand-in for a Streamlit UploadedFile."""

    def __init__(self, content: bytes):
        self._content = content

    def getvalue(self):
        return self._content


# ---------------------------------------------------------------------------
# clean_numeric_string
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        (100, 100.0),
        (12.5, 12.5),
        ("100", 100.0),
        ("$1,234.56", 1234.56),
        ("1.234,56", 1234.56),  # EU: dot=thousands, comma=decimal
        ("1,234.56", 1234.56),  # US: comma=thousands, dot=decimal
        ("12,5", 12.5),  # bare EU decimal comma
        ("1,234", 1234.0),  # bare thousands comma
        ("1.234", 1234.0),  # bare EU thousands dot
        ("1.234.567", 1234567.0),  # bare EU multi-thousands dot
        ("12.5", 12.5),  # plain US/decimal dot, unaffected
        ("99.90", 99.90),  # two trailing digits stays a decimal
        ("1.5", 1.5),  # one trailing digit stays a decimal
        ("12%", 12.0),
        ("(100)", -100.0),
        ("-50", -50.0),
        ("+50", 50.0),
        ("€ 99.90", 99.90),
        ("", None),
        ("n/a", None),
        (None, None),
    ],
)
def test_clean_numeric_string(raw, expected):
    result = clean_numeric_string(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# normalize_columns
# ---------------------------------------------------------------------------
def test_normalize_columns_names_blank_headers():
    from forecast_utils.cleaning import CleaningReport

    df = pd.DataFrame([[1, 2]], columns=["", "Unnamed: 1"])
    report = CleaningReport()
    cleaned = normalize_columns(df, report)
    assert list(cleaned.columns) == ["column_1", "column_2"]
    assert report.notes  # something was logged


def test_normalize_columns_dedupes_duplicates():
    from forecast_utils.cleaning import CleaningReport

    df = pd.DataFrame([[1, 2, 3]], columns=["date", "revenue", "revenue"])
    report = CleaningReport()
    cleaned = normalize_columns(df, report)
    assert list(cleaned.columns) == ["date", "revenue", "revenue_1"]
    assert any("Duplicate" in w for w in report.warnings)


def test_normalize_columns_trims_whitespace():
    from forecast_utils.cleaning import CleaningReport

    df = pd.DataFrame([[1]], columns=["  Date  "])
    report = CleaningReport()
    cleaned = normalize_columns(df, report)
    assert list(cleaned.columns) == ["Date"]


# ---------------------------------------------------------------------------
# drop_empty_rows_and_columns
# ---------------------------------------------------------------------------
def test_drop_empty_rows_and_columns():
    from forecast_utils.cleaning import CleaningReport

    df = pd.DataFrame(
        {
            "date": ["2024-01-01", None, "2024-01-03"],
            "value": [1, None, 3],
            "empty_col": [None, None, None],
        }
    )
    report = CleaningReport()
    cleaned = drop_empty_rows_and_columns(df, report)
    assert "empty_col" not in cleaned.columns
    assert len(cleaned) == 2
    assert any("empty column" in n for n in report.notes)
    assert any("empty row" in n for n in report.notes)


# ---------------------------------------------------------------------------
# best_effort_parse_dates
# ---------------------------------------------------------------------------
def test_best_effort_parse_dates_unambiguous_iso():
    s = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
    parsed, rate, dayfirst_used, ambiguous = best_effort_parse_dates(s)
    assert rate == 1.0
    assert not ambiguous
    assert parsed.notna().all()


def test_best_effort_parse_dates_flags_genuine_ambiguity():
    # 03/04/2024 and 04/03/2024 parse fine both ways but disagree
    s = pd.Series(["03/04/2024", "04/03/2024", "05/06/2024"])
    _, rate, _, ambiguous = best_effort_parse_dates(s)
    assert rate == 1.0
    assert ambiguous


def test_best_effort_parse_dates_unambiguous_when_only_one_side_parses():
    # "13" can't be a month, so only the day-first reading is even valid —
    # pandas' own format inference gets there before our dayfirst flag does,
    # so both branches agree on the same (correct) date.
    s = pd.Series(["13/01/2024", "14/01/2024"])
    parsed, rate, _, ambiguous = best_effort_parse_dates(s)
    assert rate == 1.0
    assert not ambiguous
    assert parsed.tolist() == [pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-14")]


# ---------------------------------------------------------------------------
# read_csv_robust
# ---------------------------------------------------------------------------
def test_read_csv_robust_sniffs_semicolon_delimiter():
    csv_text = "date;value\n2024-01-01;10\n2024-01-02;20\n"
    df, report = read_csv_robust(_FakeUpload(csv_text.encode("utf-8")))
    assert df is not None
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2
    assert any("auto-detected delimiter" in n for n in report.notes)


def test_read_csv_robust_handles_comma_normally():
    csv_text = "date,value\n2024-01-01,10\n2024-01-02,20\n"
    df, report = read_csv_robust(_FakeUpload(csv_text.encode("utf-8")))
    assert df is not None
    assert list(df.columns) == ["date", "value"]
    assert report.ok


def test_read_csv_robust_empty_file_errors():
    df, report = read_csv_robust(_FakeUpload(b""))
    assert df is None
    assert report.errors


def test_read_csv_robust_latin1_fallback():
    csv_text = "date,label\n2024-01-01,caf\xe9\n"  # 'café' in latin-1
    df, report = read_csv_robust(_FakeUpload(csv_text.encode("latin-1")))
    assert df is not None
    assert df["label"].iloc[0] == "café"
    assert any("decoded as" in n for n in report.notes)


# ---------------------------------------------------------------------------
# suggest_date_column / suggest_numeric_columns
# ---------------------------------------------------------------------------
def test_suggest_date_and_numeric_columns():
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Conversions": ["10", "20", "30"],
            "Notes": ["a", "b", "c"],
        }
    )
    assert suggest_date_column(df) == "Date"
    assert "Conversions" in suggest_numeric_columns(df, exclude=["Date"])
    assert "Notes" not in suggest_numeric_columns(df, exclude=["Date"])


# ---------------------------------------------------------------------------
# finalize_for_fit
# ---------------------------------------------------------------------------
def test_finalize_for_fit_drops_unparseable_dates_and_coerces_numbers():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04"],
            "revenue": ["$1,000.50", "200", "$3,000.00", "4000"],
        }
    )
    final_df, report = finalize_for_fit(df, "date", ["revenue"])
    assert final_df is not None
    assert len(final_df) == 3  # the bad-date row is dropped
    assert final_df["revenue"].tolist() == [1000.50, 3000.00, 4000.0]
    assert any("couldn't be parsed" in w for w in report.warnings)


def test_finalize_for_fit_gates_on_minimum_rows():
    df = pd.DataFrame({"date": ["2024-01-01"], "revenue": ["100"]})
    final_df, report = finalize_for_fit(df, "date", ["revenue"], min_rows=2)
    assert final_df is None
    assert report.errors
    assert not report.ok


def test_finalize_for_fit_reports_ambiguous_dates():
    df = pd.DataFrame(
        {
            "date": ["03/04/2024", "04/03/2024", "05/06/2024"],
            "revenue": ["100", "200", "300"],
        }
    )
    final_df, report = finalize_for_fit(df, "date", ["revenue"])
    assert final_df is not None
    assert any("ambiguous dates" in w for w in report.warnings)


def test_finalize_for_fit_flags_uncoercible_numbers_without_dropping_row():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "revenue": ["100", "not-a-number"],
        }
    )
    final_df, report = finalize_for_fit(df, "date", ["revenue"])
    assert final_df is not None
    assert len(final_df) == 2  # row kept, value just becomes NaN
    assert pd.isna(final_df["revenue"].iloc[1])
    assert any("couldn't be read as a number" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# clean_uploaded_dataframe (structural pass end-to-end)
# ---------------------------------------------------------------------------
def test_clean_uploaded_dataframe_end_to_end():
    df = pd.DataFrame(
        {
            "": [None, None, None],
            "Date ": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Revenue": ["100", "200", "300"],
        }
    )
    cleaned = clean_uploaded_dataframe(df)
    assert "column_1" not in cleaned.df.columns  # fully-empty column dropped, not renamed
    assert "Date" in cleaned.df.columns
    assert cleaned.suggested_date_col == "Date"
    assert "Revenue" in cleaned.suggested_numeric_cols


def test_clean_uploaded_dataframe_all_empty_rows_errors():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    cleaned = clean_uploaded_dataframe(df)
    assert not cleaned.report.ok
    assert cleaned.df.empty
