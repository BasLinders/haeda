"""Weather covariate: geocoding + historical/forecast pull from Open-Meteo.

Not part of FOE (the engine has no opinion on where regressor data comes
from) — this is entirely UI-repo responsibility. Once merged, weather is
just another numeric column offered in the same covariate multiselect as
any uploaded column.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests
import streamlit as st

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_DAILY_VARS = "temperature_2m_mean,precipitation_sum"
_FORECAST_HORIZON_DAYS = 16
_TIMEOUT = 15


@st.cache_data(show_spinner=False)
def geocode_location(location: str) -> dict | None:
    """Look up lat/lon for a free-text location string. Cached so repeated
    runs with the same location don't re-hit the API."""
    if not location or not location.strip():
        return None
    resp = requests.get(
        _GEOCODE_URL, params={"name": location.strip(), "count": 1}, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        return None
    hit = results[0]
    return {
        "name": hit.get("name"),
        "country": hit.get("country"),
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
    }


@st.cache_data(show_spinner=False)
def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily mean temperature + precipitation for [start_date, end_date] (YYYY-MM-DD)."""
    resp = requests.get(
        _ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": _DAILY_VARS,
            "timezone": "auto",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame(columns=["ds", "weather_temp", "weather_precipitation"])
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(daily["time"]),
            "weather_temp": daily.get("temperature_2m_mean"),
            "weather_precipitation": daily.get("precipitation_sum"),
        }
    )


@st.cache_data(show_spinner=False)
def fetch_forecast_weather(lat: float, lon: float) -> pd.DataFrame:
    """Daily forecast (~16 days ahead) for temperature + precipitation."""
    resp = requests.get(
        _FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": _DAILY_VARS,
            "forecast_days": _FORECAST_HORIZON_DAYS,
            "timezone": "auto",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame(columns=["ds", "weather_temp", "weather_precipitation"])
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(daily["time"]),
            "weather_temp": daily.get("temperature_2m_mean"),
            "weather_precipitation": daily.get("precipitation_sum"),
        }
    )


def _seasonal_average_by_doy(archive_df: pd.DataFrame) -> pd.DataFrame:
    """Average temp/precip per day-of-year, from the historical archive pull."""
    doy = archive_df.assign(doy=archive_df["ds"].dt.dayofyear)
    return doy.groupby("doy")[["weather_temp", "weather_precipitation"]].mean().reset_index()


def build_weather_covariates(
    lat: float,
    lon: float,
    history_start: dt.date,
    history_end: dt.date,
    future_dates: list,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (historical_df, future_df, regime_df).

    historical_df: ds, weather_temp, weather_precipitation — merge into the
        uploaded history by date.
    future_df: same columns covering `future_dates`, using the real Open-Meteo
        forecast where available (~16 days) and falling back to the
        historical day-of-year seasonal average further out.
    regime_df: ds, regime ("forecast" | "seasonal_average") — for showing the
        user which regime backed each future date.
    """
    archive = fetch_historical_weather(
        lat, lon, history_start.isoformat(), history_end.isoformat()
    )
    forecast = fetch_forecast_weather(lat, lon)
    forecast_dates = set(forecast["ds"].dt.date) if not forecast.empty else set()

    seasonal_avg = _seasonal_average_by_doy(archive) if not archive.empty else pd.DataFrame(
        columns=["doy", "weather_temp", "weather_precipitation"]
    )

    rows = []
    regimes = []
    for d in future_dates:
        if d in forecast_dates:
            row = forecast.loc[forecast["ds"].dt.date == d].iloc[0]
            rows.append({"ds": d, "weather_temp": row["weather_temp"], "weather_precipitation": row["weather_precipitation"]})
            regimes.append({"ds": d, "regime": "forecast"})
        else:
            doy = pd.Timestamp(d).dayofyear
            match = seasonal_avg.loc[seasonal_avg["doy"] == doy]
            if not match.empty:
                r = match.iloc[0]
                rows.append({"ds": d, "weather_temp": r["weather_temp"], "weather_precipitation": r["weather_precipitation"]})
            else:
                rows.append({"ds": d, "weather_temp": None, "weather_precipitation": None})
            regimes.append({"ds": d, "regime": "seasonal_average"})

    future_df = pd.DataFrame(rows)
    regime_df = pd.DataFrame(regimes)
    return archive, future_df, regime_df
