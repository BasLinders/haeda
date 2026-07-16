"""Robust CSV ingestion and wrangling for the forecasting app.

Every function here degrades gracefully: a messy upload should turn into a
usable DataFrame plus a plain-language report of what was fixed or dropped,
never a raw traceback. Nothing here silently changes row/column counts
without recording why in the returned report.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_CURRENCY_CHARS = "$€£¥₹"
_DATE_NAME_HINTS = ("date", "dt", "day", "period", "time", "week", "month")
_NUMERIC_NAME_HINTS = (
    "conversion", "revenue", "sales", "orders", "spend", "cost", "price",
    "amount", "qty", "quantity", "count", "sessions", "clicks", "visits",
    "value", "total",
)


@dataclass
class CleaningReport:
    """Plain-language log of every decision made while cleaning the upload."""

    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_csv_robust(uploaded_file) -> tuple[pd.DataFrame | None, CleaningReport]:
    """Read an uploaded CSV, tolerating wrong encodings and delimiters.

    `uploaded_file` is a Streamlit `UploadedFile` (file-like, supports
    `.getvalue()`); this also accepts any object with `.read()`.
    """
    report = CleaningReport()
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not raw:
        report.error("The uploaded file is empty.")
        return None, report

    text = None
    used_encoding = None
    for enc in _ENCODINGS_TO_TRY:
        try:
            text = raw.decode(enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        report.error(
            "Could not decode the file with any common encoding "
            f"({', '.join(_ENCODINGS_TO_TRY)}). Please re-save it as UTF-8 CSV."
        )
        return None, report
    if used_encoding not in ("utf-8", "utf-8-sig"):
        report.note(f"File was decoded as {used_encoding} (not UTF-8) — re-save as UTF-8 if you see garbled characters.")

    df = _try_parse_delimited(text, report)
    if df is None:
        return None, report
    return df, report


def _try_parse_delimited(text: str, report: CleaningReport) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # malformed CSV entirely
        report.error(f"Could not parse the file as CSV: {exc}")
        return None

    if df.shape[1] == 1:
        # Classic symptom of a `;`-delimited (or tab) export read with the
        # default comma separator: everything lands in one column.
        try:
            sniffed = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception:
            sniffed = None
        if sniffed is not None and sniffed.shape[1] > 1:
            report.note(
                f"File did not look comma-separated (only 1 column detected) — "
                f"auto-detected delimiter instead, found {sniffed.shape[1]} columns."
            )
            df = sniffed
        else:
            report.warn(
                "Only one column was detected. If your file uses a semicolon or "
                "tab separator this may indicate a parsing problem — check the preview below."
            )
    return df


def normalize_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Trim/collapse whitespace in headers and de-duplicate blank/duplicate names."""
    df = df.copy()
    original = list(df.columns)
    new_names = []
    seen: dict[str, int] = {}
    for i, col in enumerate(original):
        name = str(col).strip()
        name = re.sub(r"\s+", " ", name)
        if name == "" or re.match(r"^Unnamed: \d+$", name):
            name = f"column_{i + 1}"
            report.note(f"Column {i + 1} had no header — named it '{name}'.")
        if name in seen:
            seen[name] += 1
            deduped = f"{name}_{seen[name]}"
            report.warn(f"Duplicate column name '{name}' — renamed the repeat to '{deduped}'.")
            name = deduped
        else:
            seen[name] = 0
        new_names.append(name)

    renamed = {orig: new for orig, new in zip(original, new_names) if str(orig) != new}
    if renamed:
        renaming_notes = [f"'{o}' → '{n}'" for o, n in renamed.items() if str(o).strip() != n]
        if renaming_notes:
            report.note("Cleaned up column names: " + "; ".join(renaming_notes))
    df.columns = new_names
    return df


def drop_empty_rows_and_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped_cols = before_cols - df.shape[1]
    if dropped_cols:
        report.note(f"Dropped {dropped_cols} fully-empty column(s).")

    before_rows = len(df)
    df = df.dropna(axis=0, how="all")
    dropped_rows = before_rows - len(df)
    if dropped_rows:
        report.note(f"Dropped {dropped_rows} fully-empty row(s).")
    return df.reset_index(drop=True)


def _looks_like_date_name(col: str) -> bool:
    lower = col.lower()
    return any(hint in lower for hint in _DATE_NAME_HINTS)


def _looks_like_numeric_name(col: str) -> bool:
    lower = col.lower()
    return any(hint in lower for hint in _NUMERIC_NAME_HINTS)


def _date_parse_success_rate(series: pd.Series, dayfirst: bool) -> tuple[pd.Series, float]:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    non_null = series.notna().sum()
    if non_null == 0:
        return parsed, 0.0
    rate = parsed.notna().sum() / non_null
    return parsed, rate


def best_effort_parse_dates(series: pd.Series) -> tuple[pd.Series, float, bool, bool]:
    """Try both day-first and month-first parsing, return whichever parses more rows.

    The 4th return value flags genuine day/month ambiguity: both
    interpretations parse equally well but disagree on the actual date for
    at least one row (e.g. "03/04/2024" — 3 April or 4 March?), which no
    amount of heuristics can resolve from the data alone.

    Strict ISO 8601 (YYYY-MM-DD) is never ambiguous — the year always leads,
    so there's no day/month order to guess — but naively re-parsing it with
    dayfirst=True can still "succeed" by reinterpreting the month and day
    fields, which would otherwise falsely trip the ambiguity check on
    perfectly clean data. Short-circuit those out before the heuristic.
    """
    non_null = series.dropna().astype(str)
    if not non_null.empty and non_null.str.match(_ISO_DATE_RE).all():
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
        rate = parsed.notna().sum() / len(non_null)
        return parsed, rate, False, False

    parsed_mf, rate_mf = _date_parse_success_rate(series, dayfirst=False)
    parsed_df, rate_df = _date_parse_success_rate(series, dayfirst=True)

    ambiguous = False
    if rate_mf > 0 and rate_df > 0:
        both_valid = parsed_mf.notna() & parsed_df.notna()
        if both_valid.any() and not (parsed_mf[both_valid] == parsed_df[both_valid]).all():
            ambiguous = abs(rate_mf - rate_df) < 1e-9

    if rate_df > rate_mf:
        return parsed_df, rate_df, True, ambiguous
    return parsed_mf, rate_mf, False, ambiguous


def clean_numeric_string(value) -> float | None:
    """Coerce a single messy numeric-looking value to float, or None."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    for ch in _CURRENCY_CHARS:
        s = s.replace(ch, "")
    s = s.replace("%", "")
    s = s.replace(" ", " ").strip()
    s = s.replace(" ", "")

    if s.startswith("-"):
        negative = True
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    if s == "":
        return None

    has_comma = "," in s
    has_dot = "." in s
    if has_comma and has_dot:
        if s.rfind(",") > s.rfind("."):
            # European: dot = thousands, comma = decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # US: comma = thousands, dot = decimal
            s = s.replace(",", "")
    elif has_comma and not has_dot:
        # Ambiguous: "1,234" (thousands) vs "12,5" (EU decimal).
        # Treat a single comma with exactly 1-2 trailing digits as a decimal
        # separator; three trailing digits (or multiple commas) as thousands.
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    # dot-only strings are left as-is (either a plain decimal or thousands
    # dots handled by the comma/dot combo branch above)

    try:
        num = float(s)
    except ValueError:
        return None
    return -num if negative else num


def coerce_numeric_column(series: pd.Series) -> tuple[pd.Series, int]:
    """Coerce a column to numeric, tolerating currency/percent/locale formatting."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), 0
    cleaned = series.map(clean_numeric_string)
    n_failed = int(cleaned.isna().sum() - series.isna().sum())
    n_failed = max(n_failed, 0)
    return cleaned.astype(float), n_failed


def suggest_date_column(df: pd.DataFrame) -> str | None:
    best_col, best_score = None, -1.0
    for col in df.columns:
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_datetime64_any_dtype(df[col])
        ):
            _, rate, _, _ = best_effort_parse_dates(df[col])
            score = rate + (0.1 if _looks_like_date_name(col) else 0.0)
            if rate >= 0.8 and score > best_score:
                best_col, best_score = col, score
    return best_col


def suggest_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append((col, 1.0 + (0.1 if _looks_like_numeric_name(col) else 0)))
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        parsed = sample.map(clean_numeric_string)
        rate = parsed.notna().sum() / len(sample)
        if rate >= 0.8:
            candidates.append((col, rate + (0.1 if _looks_like_numeric_name(col) else 0)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates]


@dataclass
class CleanedData:
    df: pd.DataFrame
    report: CleaningReport
    suggested_date_col: str | None
    suggested_numeric_cols: list


def clean_uploaded_dataframe(df: pd.DataFrame) -> CleanedData:
    """Full structural cleaning pass: header + row/column hygiene, no type coercion yet.

    Type coercion (dates / numerics) happens per-column once the user confirms
    the column mapping, since it depends on which columns they mean as
    date/target/regressor.
    """
    report = CleaningReport()
    df = normalize_columns(df, report)
    df = drop_empty_rows_and_columns(df, report)

    if df.empty:
        report.error("No usable data rows remain after removing empty rows.")
        return CleanedData(df, report, None, [])

    suggested_date = suggest_date_column(df)
    suggested_numeric = suggest_numeric_columns(df, exclude=[suggested_date] if suggested_date else [])

    return CleanedData(df, report, suggested_date, suggested_numeric)


def finalize_for_fit(
    df: pd.DataFrame,
    date_col: str,
    numeric_cols: list,
    min_rows: int = 2,
) -> tuple[pd.DataFrame | None, CleaningReport]:
    """Coerce the chosen date + numeric columns and gate on minimum usable rows.

    Rows that fail to parse the chosen date column are dropped (with a count
    reported); numeric coercion failures become NaN in place (Prophet/pandas
    handle NaN targets/regressors at the aggregation step, but we still
    report how many were affected so the user isn't surprised).
    """
    report = CleaningReport()
    df = df.copy()

    parsed_dates, rate, dayfirst_used, ambiguous = best_effort_parse_dates(df[date_col])
    n_bad_dates = int(parsed_dates.isna().sum() - df[date_col].isna().sum())
    n_bad_dates = max(n_bad_dates, 0)
    df[date_col] = parsed_dates
    if dayfirst_used:
        report.note(f"Interpreted '{date_col}' as day-first dates (e.g. 05/03/2024 = 5 March).")
    if ambiguous:
        report.warn(
            f"'{date_col}' contains ambiguous dates (e.g. 03/04/2024 could be 3 April or "
            "4 March) — day-first and month-first parsing both succeed but disagree on some "
            "rows. Verify the parsed dates in the preview look right before trusting the forecast."
        )
    if n_bad_dates:
        report.warn(f"{n_bad_dates} row(s) had a date in '{date_col}' that couldn't be parsed and were dropped.")

    before = len(df)
    df = df.dropna(subset=[date_col])
    dropped = before - len(df)
    if dropped and dropped != n_bad_dates:
        report.note(f"Dropped {dropped} row(s) with a missing value in '{date_col}'.")

    for col in numeric_cols:
        if col not in df.columns:
            continue
        coerced, n_failed = coerce_numeric_column(df[col])
        df[col] = coerced
        if n_failed:
            report.warn(
                f"{n_failed} value(s) in '{col}' couldn't be read as a number "
                "(after stripping currency symbols/thousands separators) and were left blank."
            )

    dup_dates = int(df[date_col].duplicated().sum())
    if dup_dates:
        report.note(
            f"'{date_col}' has {dup_dates} repeated date value(s) — this is fine, they'll be "
            "aggregated together when the engine resamples to your chosen granularity."
        )

    if len(df) < min_rows:
        report.error(
            f"Only {len(df)} usable row(s) remain after cleaning — at least {min_rows} are needed to fit a forecast."
        )
        return None, report

    df = df.sort_values(date_col).reset_index(drop=True)
    return df, report
