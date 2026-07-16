import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from foe.core.models import (
    CustomHoliday,
    ForecastGranularity,
    ForecastingEngineConfig,
    GrowthMode,
    SeasonalityMode,
)
from foe.forecasting.mock_data import generate_mock_forecast_data
from foe.forecasting.operations import ForecastingEngine
from foe.viz.operations import VizEngine

from forecast_utils.cleaning import (
    CleaningReport,
    clean_numeric_string,
    clean_uploaded_dataframe,
    finalize_for_fit,
    read_csv_robust,
)
from forecast_utils.weather import build_weather_covariates, geocode_location

st.set_page_config(page_title="Demand Forecasting", page_icon="🔮", layout="wide")

st.title("🔮 Demand Forecasting")
st.write(
    "Upload historical conversions/revenue data, configure a forecast, and get a chart with "
    "an uncertainty band, cross-validation quality metrics, and an optional breakdown of what's "
    "driving the shape."
)

with st.expander("What is this, and why would I use it?"):
    st.markdown(
        """
**What it does**

Takes your historical numbers — conversions, revenue, or both, one row per day, week, or
month — and projects them forward. Instead of a single "most likely" number, you get a
realistic range of outcomes, plus an honest check (using your own past data) of how accurate
this kind of forecast tends to be for your business.

**Why use it**

- Plan budgets, staffing, or inventory around a data-backed forecast instead of a gut-feel guess.
- See how known patterns — weekend dips, a yearly holiday peak, a recurring promotion — are
  expected to play out, instead of a straight-line extrapolation of last month's numbers.
- Test "what if" scenarios by adding your own covariates (e.g. a planned marketing spend, price
  change, or weather) to see how they're expected to move the numbers.
- Get a trust signal (not just a pretty chart) so you know how much weight to put on the
  forecast before acting on it.

**What it isn't**: a crystal ball. It extrapolates from patterns in your own history — the
less history you give it, or the more your business is changing, the less reliable the
forecast gets. The tool will tell you when it doesn't have enough data to trust its own numbers.
        """
    )

with st.expander("Data requirements & CSV template"):
    st.markdown(
        """
Your CSV needs:

- **One row per date** — daily, weekly, or monthly, whichever granularity you plan to forecast at.
- **A date column** — most common formats are auto-detected; anything ambiguous gets flagged
  for you to double-check.
- **At least one number to forecast** — conversions, revenue, or both.
- *(Optional)* any other numeric columns you'd like to use as covariates later (e.g. marketing
  spend, price, foot traffic) — nothing is used automatically, you opt in further down.

Messy real-world exports are handled automatically — currency symbols, thousands separators,
semicolon delimiters, stray blank rows/columns — with a plain-language report of what was
changed, so nothing gets silently altered without you knowing.
        """
    )
    _template_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "conversions": [120, 135, 128, 142],
            "revenue": [2400.50, 2690.00, 2550.75, 2810.25],
            "marketing_spend": [500, 520, 480, 510],
        }
    )
    st.download_button(
        label="Download CSV template",
        data=_template_df.to_csv(index=False).encode("utf-8"),
        file_name="forecasting_template.csv",
        mime="text/csv",
        help=(
            "conversions/revenue are examples — only one is required, and marketing_spend is "
            "just a sample optional covariate. Replace the example rows with your own data, "
            "keeping the column headers."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_report(report: CleaningReport, title: str = "Data quality report"):
    if report.errors:
        for e in report.errors:
            st.error(e)
    if report.notes or report.warnings:
        with st.expander(
            f"{title} ({len(report.notes)} note(s), {len(report.warnings)} warning(s))",
            expanded=bool(report.warnings),
        ):
            for w in report.warnings:
                st.warning(w)
            for n in report.notes:
                st.write(f"- {n}")


@st.cache_data(show_spinner="Reading and cleaning your file...")
def load_and_clean_csv(file_bytes: bytes):
    df_raw, read_report = read_csv_robust(io.BytesIO(file_bytes))
    if df_raw is None:
        return None, read_report
    cleaned = clean_uploaded_dataframe(df_raw)
    cleaned.report.notes = read_report.notes + cleaned.report.notes
    cleaned.report.warnings = read_report.warnings + cleaned.report.warnings
    cleaned.report.errors = read_report.errors + cleaned.report.errors
    return cleaned, None


@st.cache_resource(show_spinner="Fitting forecast model(s)... this can take a moment.")
def run_fit(data: pd.DataFrame, config_json: str, future_regressors: pd.DataFrame | None):
    config = ForecastingEngineConfig.model_validate_json(config_json)
    return ForecastingEngine.fit(data, config, future_regressors)


def render_result(result, history_by_target: dict):
    st.write(result.conclusion)
    for target_name, tf in result.targets.items():
        st.divider()
        st.subheader(f"Forecast: {target_name}")

        for w in tf.warnings:
            st.warning(w)

        col1, col2, col3 = st.columns(3)
        if tf.cv_metrics is None:
            st.info(
                "Not enough history yet to check how accurate this forecast tends to be — "
                "add more historical data to see quality metrics."
            )
        else:
            col1.metric(
                "MAPE (average error %)",
                f"{tf.cv_metrics.mape:.1%}",
                help=(
                    "MAPE = Mean Absolute Percentage Error. On average, how far off past "
                    "forecasts were from what actually happened, as a percentage. Lower is "
                    "better — e.g. 10% means forecasts were typically off by about a tenth of "
                    "the actual value."
                ),
            )
            col2.metric(
                "RMSE (typical error size)",
                f"{tf.cv_metrics.rmse:.2f}",
                help=(
                    "RMSE = Root Mean Square Error. The typical size of the error in the same "
                    "units as your data (e.g. conversions or currency), with bigger misses "
                    "counted extra heavily. Lower is better; compare it to the size of your "
                    "usual numbers to judge whether it's a big deal."
                ),
            )
            col3.metric(
                "Accuracy checked over",
                f"{tf.cv_metrics.horizon_periods} period(s)",
                help=(
                    "These accuracy numbers come from replaying history: hiding the most "
                    "recent stretch of real data, forecasting it as if it were the future, and "
                    "comparing to what actually happened. This is how long that hidden stretch was."
                ),
            )

        chart_data = VizEngine.get_forecasting_chart_data(
            history_by_target[target_name], tf, target_name
        )
        fig = go.Figure()
        hist_df = pd.DataFrame(chart_data["history"])
        fc_df = pd.DataFrame(chart_data["forecast"])
        band_df = pd.DataFrame(chart_data["expected_range"])

        fig.add_trace(
            go.Scatter(
                x=band_df["ds"], y=band_df["yhat_upper"],
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_df["ds"], y=band_df["yhat_lower"],
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(99, 110, 250, 0.2)",
                name="expected range", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df["ds"], y=hist_df["y"],
                mode="markers", name="actuals",
                marker=dict(color="rgba(80, 80, 80, 0.6)", size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df["ds"], y=fc_df["yhat"],
                mode="lines", name="forecast",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.1),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(chart_data["chart_caption"])

        with st.expander("Why does the forecast look like this?"):
            for comp_name, records in tf.components.items():
                comp_df = pd.DataFrame(records)
                if comp_df.empty:
                    continue
                st.write(f"**{comp_name}**")
                st.line_chart(comp_df.set_index("ds")["value"])


# ---------------------------------------------------------------------------
# Sidebar: demo data shortcut
# ---------------------------------------------------------------------------
st.sidebar.header("Quick start")
demo_clicked = st.sidebar.button(
    "🎲 Generate demo data",
    help=(
        "Skip the upload and see the whole thing working on made-up sample data: two years "
        "of history with a growth trend, weekly and yearly patterns, some holiday spikes, and "
        "a marketing-spend covariate."
    ),
)
if demo_clicked:
    st.session_state["forecasting_mode"] = "demo"
if "forecasting_mode" not in st.session_state:
    st.session_state["forecasting_mode"] = None

if st.session_state["forecasting_mode"] == "demo":
    if st.sidebar.button("← Back to upload"):
        st.session_state["forecasting_mode"] = None
        st.rerun()

    st.info(
        "Showing a one-click demo on made-up sample data — forecasting both conversions and "
        "revenue, with custom holidays, a marketing-spend covariate, and accuracy metrics all "
        "switched on. Click 'Back to upload' in the sidebar to use your own CSV instead."
    )
    dataset = generate_mock_forecast_data(seed=42)
    result = run_fit(
        dataset.data, dataset.suggested_config.model_dump_json(), dataset.future_regressors
    )
    history_by_target = {
        target: dataset.data[["date", col]].rename(columns={"date": "ds", col: "y"})
        for target, col in [
            ("conversions", "conversions"),
            ("revenue", "revenue"),
        ]
        if target in result.targets
    }
    render_result(result, history_by_target)
    st.stop()


# ---------------------------------------------------------------------------
# Upload flow
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload historical data (CSV)",
    type=["csv"],
    help="One row per date, with at least one numeric column (conversions and/or revenue).",
)

if not uploaded:
    st.write("Or use **Generate demo data** in the sidebar for a one-click example.")
    st.stop()

cleaned, hard_error = load_and_clean_csv(uploaded.getvalue())
if cleaned is None:
    render_report(hard_error)
    st.stop()

render_report(cleaned.report, title="Cleaning report")
if not cleaned.report.ok:
    st.stop()

df = cleaned.df
with st.expander("Preview cleaned data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("1. Map your columns")
all_columns = list(df.columns)
date_col = st.selectbox(
    "Date column",
    all_columns,
    index=all_columns.index(cleaned.suggested_date_col) if cleaned.suggested_date_col in all_columns else 0,
    help="Column containing the date for each row.",
)

date_preview = pd.to_datetime(df[date_col], errors="coerce")
n_bad = int(date_preview.isna().sum() - df[date_col].isna().sum())
if n_bad == len(df):
    st.error(
        f"Could not parse any values in '{date_col}' as dates. Choose a different column."
    )
    st.stop()
elif n_bad > 0:
    st.warning(f"{n_bad} row(s) in '{date_col}' don't look like valid dates — they'll be dropped.")

col_a, col_b = st.columns(2)
with col_a:
    forecast_conversions = st.checkbox("Forecast conversions", value=True)
with col_b:
    forecast_revenue = st.checkbox("Forecast revenue", value=False)

numeric_candidates = [c for c in cleaned.suggested_numeric_cols if c != date_col] or [
    c for c in all_columns if c != date_col
]

conversions_col = None
revenue_col = None
if forecast_conversions:
    conversions_col = st.selectbox("Conversions column", numeric_candidates, key="conversions_col_select")
if forecast_revenue:
    revenue_col = st.selectbox("Revenue column", numeric_candidates, key="revenue_col_select")

if not forecast_conversions and not forecast_revenue:
    st.error("Select at least one of 'Forecast conversions' / 'Forecast revenue'.")
    st.stop()
if conversions_col and revenue_col and conversions_col == revenue_col:
    st.error("Conversions and revenue must be different columns.")
    st.stop()

st.subheader("2. Covariates (optional)")
st.caption(
    "A covariate is anything else that might explain the ups and downs in your numbers — "
    "like a price change, a marketing campaign, or the weather. Adding one can make the "
    "forecast more accurate if it genuinely affects your numbers."
)
non_target_cols = [c for c in all_columns if c not in {date_col, conversions_col, revenue_col}]
weather_enabled = st.checkbox(
    "Add weather as a covariate",
    help=(
        "Looks up the location you type in below, fetches what the weather actually was on "
        "each historical date, and predicts what it will be for your forecast period (using "
        "a real weather forecast for the next couple of weeks, and a typical-for-the-season "
        "estimate further out than that)."
    ),
)
weather_location = None
geocoded = None
if weather_enabled:
    weather_location = st.text_input(
        "Location (city, or 'city, country')",
        key="weather_location",
        help="Type the city your data is about, e.g. 'Amsterdam' or 'Amsterdam, Netherlands'.",
    )
    if weather_location:
        try:
            geocoded = geocode_location(weather_location)
        except Exception as exc:
            st.error(f"Could not look up that location: {exc}")
        if geocoded is None and weather_location:
            st.warning("Location not found — try a more specific name.")
        elif geocoded:
            st.caption(f"Using weather for: {geocoded['name']}, {geocoded['country']}")

available_regressors = list(non_target_cols)
if geocoded:
    available_regressors += ["weather_temp", "weather_precipitation"]

regressors = st.multiselect(
    "Covariates to include",
    available_regressors,
    help=(
        "Pick any columns from your file (plus weather, if you enabled it above) that you "
        "think help explain your numbers. Nothing is added automatically — you choose. Good "
        "candidates: price changes, marketing spend, weather (temperature/rainfall), the mix "
        "of organic vs. paid traffic, or competitor promotions."
    ),
)

st.subheader("3. Forecast settings")
granularity_label = st.radio(
    "Granularity",
    ["daily", "weekly", "monthly"],
    horizontal=True,
    help=(
        "How your data gets grouped before the model looks at it: one point per day, per "
        "week, or per month. Your daily ups-and-downs pattern (e.g. weekends being quieter) "
        "only makes sense at daily granularity — grouping into weeks or months averages that "
        "pattern away, so it's dropped automatically at those settings."
    ),
)
history_span_days = (pd.to_datetime(df[date_col], errors="coerce").max() - pd.to_datetime(df[date_col], errors="coerce").min()).days
if granularity_label == "monthly" and history_span_days < 700:
    st.warning(
        "Monthly granularity with under ~2 years of history can make cross-validation and "
        "yearly seasonality unreliable — the forecast may still run, but treat it cautiously."
    )

periods = st.number_input(
    "How many days/weeks/months ahead to forecast",
    min_value=1,
    value=30,
    step=1,
    help="Uses whatever granularity you picked above — e.g. 30 means 30 days if daily, 30 weeks if weekly.",
)

growth_logistic = st.toggle(
    "Use logistic growth (has a ceiling)",
    value=False,
    help=(
        "Logistic growth assumes there's a ceiling your numbers can't exceed (e.g. venue "
        "capacity). Only use this if you have a concrete cap in mind — otherwise linear is safer."
    ),
)
cap = floor = None
if growth_logistic:
    target_col_for_cap = conversions_col or revenue_col
    numeric_preview = df[target_col_for_cap].map(clean_numeric_string).dropna()
    default_cap = float(numeric_preview.max()) * 1.5 if not numeric_preview.empty else 100.0
    cap = st.number_input(
        "Cap (required)",
        value=default_cap,
        help="The highest value your numbers could ever realistically reach — the ceiling the forecast will level off toward.",
    )
    use_floor = st.checkbox("Set a floor too", value=False)
    if use_floor:
        floor = st.number_input(
            "Floor",
            value=0.0,
            help="The lowest value your numbers could ever realistically fall to.",
        )

seasonality_multiplicative = st.toggle(
    "Multiplicative seasonality",
    value=True,
    help=(
        "Multiplicative: seasonal swings grow proportionally with the trend (e.g. a growing "
        "business's summer peak grows too). Additive: seasonal swings stay a constant absolute "
        "size regardless of trend."
    ),
)

with st.expander("Advanced: how confident should the forecast's shaded range be?"):
    st.caption(
        "These two settings are optional — the defaults work well for most cases."
    )
    interval_width = st.slider(
        "How wide should the 'expected range' band be?",
        min_value=0.5,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help=(
            "How sure the model should be that the real future value lands inside the shaded "
            "band on the chart. 0.95 means 'the model expects the actual number to fall in "
            "this range 95% of the time' — a wider range plays it safer but is less precise."
        ),
    )
    custom_cv = st.checkbox(
        "Check accuracy over a different time span than the forecast horizon",
        value=False,
        help=(
            "By default, accuracy (MAPE/RMSE below) is checked by replaying history over a "
            "span as long as your forecast horizon. Turn this on to check it over a shorter or "
            "longer span instead."
        ),
    )
    cv_horizon_periods = None
    if custom_cv:
        cv_horizon_periods = st.number_input(
            "Span to check accuracy over (in the chosen granularity's units)",
            min_value=1,
            value=int(periods),
        )

st.subheader("4. Custom holidays / events (optional)")
st.caption(
    "Tell the model about specific dates that broke your normal pattern — a holiday, a big "
    "promotion, a store closure — so it treats them as one-off exceptions instead of assuming "
    "they represent a new normal. Matched to the exact date you enter: if you picked weekly or "
    "monthly grouping above, make the window (the days before/after) wide enough to reach into "
    "the week or month it falls in, or the model won't notice it."
)
holidays_df = st.data_editor(
    pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"]),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "holiday": st.column_config.TextColumn("Name", help="A short label, e.g. 'Black Friday' or 'Store closed'."),
        "ds": st.column_config.DateColumn("Date", help="The exact date the event happened."),
        "lower_window": st.column_config.NumberColumn(
            "Days before it also affected",
            step=1,
            help="Use a negative number to also flag days leading up to the event, e.g. -2 for two days before. Use 0 for just the date itself.",
        ),
        "upper_window": st.column_config.NumberColumn(
            "Days after it also affected",
            step=1,
            help="Use a positive number to also flag days after the event, e.g. 2 for two days after. Use 0 for just the date itself.",
        ),
    },
)

# Future regressor values are collected here, *before* the Run button, so the
# user actually gets a chance to edit them — a data_editor rendered inside the
# button's own gated block would only ever show its default prefill, since
# the fit call runs in that same script pass before any edit could come back.
weather_cols = {"weather_temp", "weather_precipitation"} & set(regressors)
uploaded_regressor_cols = [r for r in regressors if r not in weather_cols]
future_regressors_df = None
future_dates = None

if regressors:
    step = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}[granularity_label]
    raw_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    last_ds = pd.Series(0, index=raw_dates).resample(step).sum().index.max()
    future_dates = pd.date_range(start=last_ds, periods=int(periods) + 1, freq=step)[1:]
    future_parts = {"ds": future_dates}

    if weather_cols and geocoded:
        history_start = raw_dates.min().date()
        history_end = raw_dates.max().date()
        archive_df, future_weather_df, regime_df = build_weather_covariates(
            geocoded["latitude"], geocoded["longitude"], history_start, history_end,
            [d.date() for d in future_dates],
        )
        for col in weather_cols:
            future_parts[col] = future_weather_df[col].values
        with st.expander("Where did the future weather numbers come from?"):
            st.caption(
                "'forecast' = an actual weather forecast for that date. 'seasonal_average' = "
                "no forecast reaches that far out yet, so it uses the typical weather for that "
                "time of year based on your location's history instead."
            )
            st.dataframe(regime_df, use_container_width=True)

    if uploaded_regressor_cols:
        st.subheader("5. Future values for your uploaded covariates")
        st.caption(
            "The engine needs a value for each selected covariate across the whole forecast "
            "horizon (e.g. a marketing-spend plan). Pre-filled with the last known value — edit "
            "with your actual plan if you have one."
        )
        prefill = {"ds": future_dates}
        for col in uploaded_regressor_cols:
            coerced_col = df[col].map(clean_numeric_string).dropna()
            last_val = float(coerced_col.iloc[-1]) if len(coerced_col) else 0.0
            prefill[col] = [last_val] * len(future_dates)
        future_plan_df = st.data_editor(
            pd.DataFrame(prefill),
            use_container_width=True,
            key="future_regressor_editor",
            column_config={"ds": st.column_config.DateColumn("Date", disabled=True)},
        )
        for col in uploaded_regressor_cols:
            future_parts[col] = future_plan_df[col].values

    future_regressors_df = pd.DataFrame(future_parts)

st.divider()
run_clicked = st.button("Run forecast", type="primary")

if run_clicked:
    numeric_cols_to_coerce = [c for c in [conversions_col, revenue_col] if c] + [
        r for r in regressors if r in non_target_cols
    ]
    final_df, finalize_report = finalize_for_fit(df, date_col, numeric_cols_to_coerce)
    render_report(finalize_report, title="Final cleaning before fit")

    if final_df is None:
        st.session_state["forecast_result"] = None
    else:
        holidays = []
        holiday_error = False
        for _, row in holidays_df.dropna(subset=["holiday", "ds"]).iterrows():
            try:
                holidays.append(
                    CustomHoliday(
                        holiday=str(row["holiday"]),
                        ds=pd.to_datetime(row["ds"]).date(),
                        lower_window=int(row.get("lower_window") or 0),
                        upper_window=int(row.get("upper_window") or 0),
                    )
                )
            except ValidationError as exc:
                st.error(f"Invalid holiday row '{row.get('holiday')}': {exc}")
                holiday_error = True

        if holiday_error:
            st.session_state["forecast_result"] = None
        else:
            try:
                config = ForecastingEngineConfig(
                    date_col=date_col,
                    conversions_col=conversions_col,
                    revenue_col=revenue_col,
                    granularity=ForecastGranularity(granularity_label),
                    periods=int(periods),
                    growth=GrowthMode.LOGISTIC if growth_logistic else GrowthMode.LINEAR,
                    cap=cap,
                    floor=floor,
                    seasonality_mode=(
                        SeasonalityMode.MULTIPLICATIVE if seasonality_multiplicative else SeasonalityMode.ADDITIVE
                    ),
                    holidays=holidays,
                    regressors=regressors,
                    interval_width=interval_width,
                    cv_horizon_periods=cv_horizon_periods,
                )
            except ValidationError as exc:
                st.error(str(exc))
                st.session_state["forecast_result"] = None
            else:
                if weather_cols and geocoded:
                    history_start = pd.to_datetime(final_df[date_col]).min().date()
                    history_end = pd.to_datetime(final_df[date_col]).max().date()
                    archive_df, _, _ = build_weather_covariates(
                        geocoded["latitude"], geocoded["longitude"], history_start, history_end, []
                    )
                    final_df = final_df.merge(
                        archive_df.rename(columns={"ds": date_col}), how="left", on=date_col
                    )

                fit_key_df = final_df[
                    [date_col] + [c for c in [conversions_col, revenue_col] if c] + regressors
                ]
                try:
                    result = run_fit(fit_key_df, config.model_dump_json(), future_regressors_df)
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state["forecast_result"] = None
                else:
                    history_by_target = {}
                    if conversions_col:
                        history_by_target["conversions"] = final_df[[date_col, conversions_col]].rename(
                            columns={date_col: "ds", conversions_col: "y"}
                        )
                    if revenue_col:
                        history_by_target["revenue"] = final_df[[date_col, revenue_col]].rename(
                            columns={date_col: "ds", revenue_col: "y"}
                        )
                    st.session_state["forecast_result"] = result
                    st.session_state["forecast_history"] = history_by_target

if st.session_state.get("forecast_result") is not None:
    render_result(st.session_state["forecast_result"], st.session_state["forecast_history"])
