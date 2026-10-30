import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_COLUMNS = ["account_age_days", "trades_per_day", "baseline_trades_per_day",
               "linked_account_count", "timing_regularity"]
FEATURES = ["account_age_days", "trades_per_day", "spike_ratio",
            "linked_account_count", "timing_regularity"]
LOG_COLS = ["trades_per_day", "spike_ratio"]


def build_features(raw):
    """Raw account table -> the 5 features, in original units (not scaled)."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = raw.copy()
    baseline = df["baseline_trades_per_day"].replace(0, np.nan)
    df["spike_ratio"] = (df["trades_per_day"] / baseline).fillna(1.0)

    if "account_id" in df.columns:
        df = df.set_index("account_id")
    return df[FEATURES]


def fit_scaler(features):
    """Log the heavy-tailed columns, then fit the StandardScaler."""
    x = features.copy()
    x[LOG_COLS] = np.log1p(x[LOG_COLS])
    return StandardScaler().fit(x[FEATURES])


def prepare_for_model(features, scaler):
    """Log the heavy-tailed columns, then apply an already-fitted scaler."""
    x = features.copy()
    x[LOG_COLS] = np.log1p(x[LOG_COLS])
    scaled = scaler.transform(x[FEATURES])
    return pd.DataFrame(scaled, index=x.index, columns=FEATURES)


if __name__ == "__main__":
    # Rebuilds all Phase 4 files from data/features.csv (run from the repo root)
    import json, joblib
    raw = pd.read_csv("data/features.csv")
    features = build_features(raw)
    scaler = fit_scaler(features)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    json.dump(FEATURES, open("artifacts/feature_list.json", "w"))
    prepare_for_model(features, scaler).to_csv("data/feature_matrix.csv")
    print("Saved scaler, feature list and feature matrix.")