import pandas as pd
import requests
from pathlib import Path

# Farm / Summerville, OR area
LAT = 45.4737
LON = -117.9898

START_DATE = "2020-06-14"
END_DATE = "2026-06-14"

OUTPUT_PATH = Path("../data/weather.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,

    # Daily variables available directly from Open-Meteo
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "et0_fao_evapotranspiration",
    ],

    # Hourly variables we aggregate to daily values ourselves
    "hourly": [
        "relative_humidity_2m",
        "wind_speed_10m",
    ],

    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()
data = response.json()

# -----------------------------
# Build daily dataframe
# -----------------------------

daily = pd.DataFrame(data["daily"])
daily["date"] = pd.to_datetime(daily["time"]).dt.date

daily = daily.rename(
    columns={
        "temperature_2m_max": "max_temp_f",
        "temperature_2m_min": "min_temp_f",
        "temperature_2m_mean": "mean_temp_f",
        "precipitation_sum": "precip_in",
        "wind_speed_10m_max": "max_wind_mph",
        "et0_fao_evapotranspiration": "et0_in",
    }
)

daily = daily.drop(columns=["time"])

# -----------------------------
# Build hourly dataframe
# -----------------------------

hourly = pd.DataFrame(data["hourly"])
hourly["time"] = pd.to_datetime(hourly["time"])
hourly["date"] = hourly["time"].dt.date

hourly_daily = (
    hourly.groupby("date", as_index=False)
    .agg(
        mean_humidity_pct=("relative_humidity_2m", "mean"),
        avg_wind_mph=("wind_speed_10m", "mean"),
    )
)

# -----------------------------
# Merge daily + hourly aggregates
# -----------------------------

weather = daily.merge(hourly_daily, on="date", how="left")

# -----------------------------
# Add engineered variables
# -----------------------------

weather = weather.sort_values("date").reset_index(drop=True)

# Rolling temperature features
weather["temp_3day_avg_f"] = (
    weather["mean_temp_f"]
    .rolling(window=3, min_periods=1)
    .mean()
)

weather["temp_5day_avg_f"] = (
    weather["mean_temp_f"]
    .rolling(window=5, min_periods=1)
    .mean()
)

# Rolling rainfall features
weather["precip_3day_sum_in"] = (
    weather["precip_in"]
    .rolling(window=3, min_periods=1)
    .sum()
)

weather["precip_5day_sum_in"] = (
    weather["precip_in"]
    .rolling(window=5, min_periods=1)
    .sum()
)

# Rolling wind feature
weather["wind_3day_avg_mph"] = (
    weather["avg_wind_mph"]
    .rolling(window=3, min_periods=1)
    .mean()
)

# Rolling evapotranspiration feature
weather["et0_3day_sum_in"] = (
    weather["et0_in"]
    .rolling(window=3, min_periods=1)
    .sum()
)

# Days since measurable rain
# You can adjust this threshold if needed.
RAIN_THRESHOLD_IN = 0.01

days_since_rain = []
last_rain_index = None

for idx, precip in enumerate(weather["precip_in"]):
    if pd.notna(precip) and precip >= RAIN_THRESHOLD_IN:
        last_rain_index = idx
        days_since_rain.append(0)
    elif last_rain_index is None:
        days_since_rain.append(pd.NA)
    else:
        days_since_rain.append(idx - last_rain_index)

weather["days_since_rain"] = days_since_rain

# Date / seasonality features
weather["date"] = pd.to_datetime(weather["date"])
weather["day_of_year"] = weather["date"].dt.dayofyear
weather["month"] = weather["date"].dt.month

# Keep date as YYYY-MM-DD string for clean CSV output
weather["date"] = weather["date"].dt.strftime("%Y-%m-%d")

# -----------------------------
# Order columns to match README
# -----------------------------

column_order = [
    "date",
    "max_temp_f",
    "min_temp_f",
    "mean_temp_f",
    "mean_humidity_pct",
    "precip_in",
    "avg_wind_mph",
    "max_wind_mph",
    "et0_in",
    "temp_3day_avg_f",
    "temp_5day_avg_f",
    "precip_3day_sum_in",
    "precip_5day_sum_in",
    "wind_3day_avg_mph",
    "et0_3day_sum_in",
    "days_since_rain",
    "day_of_year",
    "month",
]

weather = weather[column_order]

# Round numeric columns for cleaner student-facing data
numeric_cols = weather.select_dtypes(include="number").columns
weather[numeric_cols] = weather[numeric_cols].round(3)

# Save
weather.to_csv(OUTPUT_PATH, index=False)

print(f"Saved weather data to {OUTPUT_PATH}")
print(f"Rows: {len(weather)}")
weather.head()