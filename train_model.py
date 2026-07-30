"""
train_model.py
---------------
Loads load_data.csv, builds time-based features, trains a
RandomForestRegressor to forecast electricity load, evaluates it,
and saves the trained model + feature list for the Streamlit app.

Run:
    python3 train_model.py
Produces:
    model.joblib
    feature_importance.png
    actual_vs_predicted.png
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split




FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_c", "load_lag_1", "load_lag_24",
]
TARGET = "load_mw"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features: load 1 hour ago, and load same hour yesterday
    df["load_lag_1"] = df["load_mw"].shift(1)
    df["load_lag_24"] = df["load_mw"].shift(24)

    df = df.dropna().reset_index(drop=True)
    return df


def main():
    df = pd.read_csv("load_data.csv")
    df = build_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    # Time-ordered split (don't shuffle time series data!)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"MAE:  {mae:.3f} MW")
    print(f"RMSE: {rmse:.3f} MW")
    print(f"R2:   {r2:.3f}")

    # Save model + feature list together
    joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
    print("Saved model.joblib")

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(6, 4))
    importances.plot(kind="barh")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()

    # Actual vs predicted (last 7 days of test set)
    plt.figure(figsize=(10, 4))
    test_dates = df["datetime"].iloc[split_idx:].values
    n_show = 24 * 7
    plt.plot(test_dates[-n_show:], y_test.values[-n_show:], label="Actual")
    plt.plot(test_dates[-n_show:], preds[-n_show:], label="Predicted", alpha=0.8)
    plt.legend()
    plt.title("Actual vs Predicted Load (last 7 days of test set)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120)
    plt.close()
    print("Saved feature_importance.png and actual_vs_predicted.png")


if __name__ == "__main__":
    main()
