"""
data_generator.py
------------------
Generates a realistic synthetic hourly electricity load dataset.

Why synthetic data?
- Lets you build and test the whole pipeline immediately, with no
  download/signup needed.
- Once your pipeline works, you can swap this out for a real dataset,
  e.g.:
    - UCI "Individual household electric power consumption"
    - Kaggle "Hourly Energy Consumption" (PJM Interconnection)
  As long as the real dataset has a datetime column and a load/demand
  column, the rest of this project (train_model.py, app.py) will work
  with minimal changes.

Run:
    python3 data_generator.py
Produces:
    load_data.csv
"""

import numpy as np
import pandas as pd

def generate_load_data(start="2023-01-01", periods_days=365, freq_per_day=24, seed=42):
    rng = np.random.default_rng(seed)
    n = periods_days * freq_per_day
    timestamps = pd.date_range(start=start, periods=n, freq="h")

    hour = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values      # 0=Mon ... 6=Sun
    day_of_year = timestamps.dayofyear.values

    # --- Base daily pattern: low at night, peaks morning & evening ---
    daily_pattern = (
        8
        + 4 * np.sin((hour - 6) / 24 * 2 * np.pi)      # broad daily wave
        + 3 * np.exp(-((hour - 9) ** 2) / 8)           # morning peak ~9am
        + 5 * np.exp(-((hour - 19) ** 2) / 10)         # evening peak ~7pm
    )

    # --- Weekly pattern: weekends have lower industrial/commercial load ---
    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)

    # --- Seasonal pattern: higher load in summer (cooling) & winter (heating) ---
    seasonal_pattern = 3 * np.sin((day_of_year / 365) * 2 * np.pi + np.pi / 2) ** 2

    # --- Synthetic temperature (Celsius), correlated with season ---
    temperature = (
        25
        + 8 * np.sin((day_of_year / 365) * 2 * np.pi - np.pi / 2)
        + rng.normal(0, 2, n)
    )

    # Temperature effect on load: extra load when it's very hot (AC) or very cold (heating)
    temp_effect = 0.08 * (temperature - 22) ** 2 / 10

    # --- Combine + noise ---
    load = (
        daily_pattern * weekend_factor
        + seasonal_pattern
        + temp_effect
        + rng.normal(0, 0.6, n)
    )
    load = np.clip(load, 2, None)  # load can't be negative

    df = pd.DataFrame({
        "datetime": timestamps,
        "load_mw": np.round(load, 3),
        "temperature_c": np.round(temperature, 2),
    })
    return df


if __name__ == "__main__":
    df = generate_load_data()
    df.to_csv("load_data.csv", index=False)
    print(f"Generated {len(df)} rows -> load_data.csv")
    print(df.head())
