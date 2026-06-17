"""
Build a clean machine-learning dataset for the farm irrigation workshop.

Run from the repo root:

    python scripts/build_model_dataset.py

Inputs:
    data/weather.csv
    data/irrigation.csv        optional

Outputs:
    data/model_dataset.csv
    data/model_metadata.json

If real farmer irrigation data is available, the model target is:

    irrigation_amount

If farmer irrigation data is not available, we create a proxy target:

    estimated_irrigation_need_in = max(et0_in - precip_in, 0)

This proxy means:
    estimated water need = water lost to weather - water added by rain
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


DATA_DIR = Path("data")

WEATHER_PATH = DATA_DIR / "weather.csv"
IRRIGATION_PATH = DATA_DIR / "irrigation.csv"

OUTPUT_DATASET_PATH = DATA_DIR / "model_dataset.csv"
OUTPUT_METADATA_PATH = DATA_DIR / "model_metadata.json"


def load_weather() -> pd.DataFrame:
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(f"Could not find {WEATHER_PATH}")

    weather = pd.read_csv(WEATHER_PATH)

    if "date" not in weather.columns:
        raise ValueError("weather.csv must contain a 'date' column")

    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def load_irrigation_if_available() -> pd.DataFrame | None:
    if not IRRIGATION_PATH.exists():
        return None

    irrigation = pd.read_csv(IRRIGATION_PATH)

    if "date" not in irrigation.columns:
        raise ValueError("irrigation.csv must contain a 'date' column")

    if "irrigation_amount" not in irrigation.columns:
        raise ValueError("irrigation.csv must contain an 'irrigation_amount' column")

    irrigation["date"] = pd.to_datetime(irrigation["date"])
    return irrigation


def add_missing_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure required weather columns exist.
    This keeps the pipeline from breaking if a source file is incomplete.
    """

    required_cols = [
        "max_temp_f",
        "min_temp_f",
        "mean_temp_f",
        "mean_humidity_pct",
        "precip_in",
        "avg_wind_mph",
        "max_wind_mph",
        "et0_in",
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: missing {col}. Filling with 0.")
            df[col] = 0.0

    if "mean_temp_f" not in df.columns:
        df["mean_temp_f"] = (df["max_temp_f"] + df["min_temp_f"]) / 2

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    df["temp_3day_avg_f"] = df["mean_temp_f"].rolling(3, min_periods=1).mean()
    df["temp_5day_avg_f"] = df["mean_temp_f"].rolling(5, min_periods=1).mean()

    df["precip_3day_sum_in"] = df["precip_in"].rolling(3, min_periods=1).sum()
    df["precip_5day_sum_in"] = df["precip_in"].rolling(5, min_periods=1).sum()

    df["wind_3day_avg_mph"] = df["avg_wind_mph"].rolling(3, min_periods=1).mean()

    df["et0_3day_sum_in"] = df["et0_in"].rolling(3, min_periods=1).sum()

    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    return df


def add_days_since_rain(df: pd.DataFrame, rain_threshold_in: float = 0.01) -> pd.DataFrame:
    days_since_rain = []
    last_rain_index = None

    for i, precip in enumerate(df["precip_in"].fillna(0)):
        if precip >= rain_threshold_in:
            last_rain_index = i
            days_since_rain.append(0)
        elif last_rain_index is None:
            days_since_rain.append(i + 1)
        else:
            days_since_rain.append(i - last_rain_index)

    df["days_since_rain"] = days_since_rain
    return df


def add_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple proxy target when farmer irrigation data is unavailable.

    estimated_irrigation_need_in = max(et0_in - precip_in, 0)
    """

    df["estimated_irrigation_need_in"] = (
        df["et0_in"].fillna(0) - df["precip_in"].fillna(0)
    ).clip(lower=0)

    df["irrigation_amount"] = df["estimated_irrigation_need_in"]
    df["irrigation_unit"] = "proxy_inches"
    df["target_type"] = "proxy_et0_minus_precip"

    return df


def add_irrigation_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    df["previous_irrigation_amount"] = df["irrigation_amount"].shift(1).fillna(0)

    days_since_irrigation = []
    last_irrigation_index = None

    for i, amount in enumerate(df["irrigation_amount"].fillna(0)):
        if amount > 0.001:
            last_irrigation_index = i
            days_since_irrigation.append(0)
        elif last_irrigation_index is None:
            days_since_irrigation.append(i + 1)
        else:
            days_since_irrigation.append(i - last_irrigation_index)

    df["days_since_irrigation"] = days_since_irrigation

    return df


def get_feature_columns(using_proxy_target: bool) -> list[str]:
    """
    If the target is created from et0_in, we do NOT include et0_in as a feature.
    That would leak the answer into the model.
    """

    feature_cols = [
        "max_temp_f",
        "min_temp_f",
        "mean_temp_f",
        "mean_humidity_pct",
        "precip_in",
        "avg_wind_mph",
        "max_wind_mph",
        "temp_3day_avg_f",
        "temp_5day_avg_f",
        "precip_3day_sum_in",
        "precip_5day_sum_in",
        "wind_3day_avg_mph",
        "days_since_rain",
        "previous_irrigation_amount",
        "days_since_irrigation",
        "day_of_year",
        "month",
    ]

    if not using_proxy_target:
        feature_cols.extend([
            "et0_in",
            "et0_3day_sum_in",
        ])

    return feature_cols


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    weather = load_weather()
    irrigation = load_irrigation_if_available()

    if irrigation is not None:
        print("Found farmer irrigation data. Merging weather.csv + irrigation.csv.")

        df = weather.merge(irrigation, on="date", how="inner")

        if "irrigation_unit" not in df.columns:
            df["irrigation_unit"] = "unknown"

        df["target_type"] = "real_farmer_irrigation"
        using_proxy_target = False

    else:
        print("No irrigation.csv found. Creating proxy irrigation target from ET0 - rain.")

        df = weather.copy()
        df = add_proxy_target(df)
        using_proxy_target = True

    df = add_missing_basic_columns(df)
    df = add_weather_features(df)
    df = add_days_since_rain(df)
    df = add_irrigation_history_features(df)

    target_col = "irrigation_amount"
    feature_cols = get_feature_columns(using_proxy_target)

    keep_cols = [
        "date",
        "target_type",
        "irrigation_amount",
        "irrigation_unit",
    ]

    if "estimated_irrigation_need_in" in df.columns:
        keep_cols.append("estimated_irrigation_need_in")

    optional_cols = [
        "field_name",
        "notes",
        "pivot_runtime_hours",
        "pivot_speed_pct",
    ]

    keep_cols.extend([col for col in optional_cols if col in df.columns])
    keep_cols.extend(feature_cols)

    keep_cols = list(dict.fromkeys(keep_cols))

    model_df = df[keep_cols].copy()

    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    numeric_cols = model_df.select_dtypes(include="number").columns
    model_df[numeric_cols] = model_df[numeric_cols].round(4)

    model_df["date"] = pd.to_datetime(model_df["date"]).dt.strftime("%Y-%m-%d")

    model_df.to_csv(OUTPUT_DATASET_PATH, index=False)

    metadata = {
        "target_col": target_col,
        "target_type": model_df["target_type"].iloc[0],
        "feature_cols": feature_cols,
        "using_proxy_target": using_proxy_target,
        "rows": len(model_df),
        "date_min": model_df["date"].min(),
        "date_max": model_df["date"].max(),
        "notes": (
            "If using_proxy_target is true, irrigation_amount is a proxy equal to "
            "max(et0_in - precip_in, 0), not real farmer irrigation."
        ),
    }

    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved dataset: {OUTPUT_DATASET_PATH}")
    print(f"Saved metadata: {OUTPUT_METADATA_PATH}")
    print(f"Rows: {len(model_df)}")
    print(f"Date range: {metadata['date_min']} to {metadata['date_max']}")
    print(f"Target type: {metadata['target_type']}")
    print()
    print("Feature columns:")
    for col in feature_cols:
        print(f"  - {col}")


if __name__ == "__main__":
    main()