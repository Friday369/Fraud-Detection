"""Generate simulated trading accounts: features (what models see) and labels (evaluation only)."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = ["account_id", "account_age_days", "trades_per_day",
                "baseline_trades_per_day", "linked_account_count",
                "timing_regularity", "alert_source_count"]
LABEL_COLS = ["account_id", "is_anomaly", "anomaly_type", "difficulty"]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def split_counts(n, subtle_fraction):
    """Split n accounts into (n_obvious, n_subtle)."""
    n_subtle = int(round(n * subtle_fraction))
    return n - n_subtle, n_subtle


def generate_normal_accounts(n, rng):
    """Create n normal (non-abusive) trading accounts."""
    account_age_days = rng.integers(1, 901, size=n)
    baseline = rng.lognormal(mean=np.log(8), sigma=0.8, size=n)
    trades_per_day = baseline * rng.lognormal(mean=0, sigma=0.2, size=n)
    linked_account_count = rng.choice([0, 1, 2], size=n, p=[0.75, 0.22, 0.03])
    timing_regularity = np.clip(rng.normal(loc=1.0, scale=0.25, size=n), 0.45, 2.0)
    alert_source_count = rng.choice([1, 2], size=n, p=[0.7, 0.3])

    return pd.DataFrame({
        "account_age_days": account_age_days,
        "trades_per_day": trades_per_day.round(2),
        "baseline_trades_per_day": baseline.round(2),
        "linked_account_count": linked_account_count,
        "timing_regularity": timing_regularity.round(3),
        "alert_source_count": alert_source_count,
    })


def generate_trial_abuse(n, subtle_fraction, rng):
    """Brand-new accounts trading far above their own baseline."""
    n_obvious, n_subtle = split_counts(n, subtle_fraction)
    df = generate_normal_accounts(n, rng)
    multiplier = np.concatenate([rng.uniform(10, 30, size=n_obvious),
                                 rng.uniform(3, 5, size=n_subtle)])
    df["account_age_days"] = rng.integers(0, 4, size=n)
    df["trades_per_day"] = (df["baseline_trades_per_day"] * multiplier).round(2)
    df["difficulty"] = ["obvious"] * n_obvious + ["subtle"] * n_subtle
    df["anomaly_type"] = "trial_abuse"
    return df


def generate_multi_account(n, subtle_fraction, rng):
    """Accounts sharing a device/IP with many other accounts."""
    n_obvious, n_subtle = split_counts(n, subtle_fraction)
    df = generate_normal_accounts(n, rng)
    df["linked_account_count"] = np.concatenate([rng.integers(6, 13, size=n_obvious),
                                                 rng.integers(3, 5, size=n_subtle)])
    df["difficulty"] = ["obvious"] * n_obvious + ["subtle"] * n_subtle
    df["anomaly_type"] = "multi_account"
    return df


def generate_bots(n, subtle_fraction, rng):
    """Accounts trading with machine-like, very regular timing."""
    n_obvious, n_subtle = split_counts(n, subtle_fraction)
    df = generate_normal_accounts(n, rng)
    df["timing_regularity"] = np.concatenate([rng.uniform(0.01, 0.10, size=n_obvious),
                                              rng.uniform(0.30, 0.44, size=n_subtle)]).round(3)
    df["difficulty"] = ["obvious"] * n_obvious + ["subtle"] * n_subtle
    df["anomaly_type"] = "bot"
    return df


def validate_features(features):
    """Raise ValueError if the features table is malformed."""
    if list(features.columns) != FEATURE_COLS:
        raise ValueError(f"Unexpected columns: {list(features.columns)}")
    if features.isna().any().any():
        raise ValueError("Features contain missing values")
    if not features["account_id"].is_unique:
        raise ValueError("account_id must be unique")

    ranges = {  # column: (min allowed, max allowed)
        "account_age_days": (0, 900),
        "trades_per_day": (0, 10_000),
        "baseline_trades_per_day": (0.01, 10_000),   # must be > 0
        "linked_account_count": (0, 50),
        "timing_regularity": (0, 5),
        "alert_source_count": (1, 4),
    }
    for col, (lo, hi) in ranges.items():
        if not pd.api.types.is_numeric_dtype(features[col]):
            raise ValueError(f"{col} must be numeric")
        if features[col].min() < lo or features[col].max() > hi:
            raise ValueError(f"{col} has values outside [{lo}, {hi}]")


def generate_dataset(config, seed=None):
    """Return (features, labels). Pass a seed to override the one in config."""
    seed = config["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    n_normal = config["accounts"]["n_normal"]
    n_abuse = config["accounts"]["n_per_abuse_type"]
    frac = config["accounts"]["subtle_fraction"]

    normal = generate_normal_accounts(n_normal, rng)
    normal["anomaly_type"] = "normal"
    normal["difficulty"] = "none"

    parts = [normal,
             generate_trial_abuse(n_abuse, frac, rng),
             generate_multi_account(n_abuse, frac, rng),
             generate_bots(n_abuse, frac, rng)]
    df = pd.concat(parts, ignore_index=True)

    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)   # shuffle, seeded
    df["account_id"] = [f"ACC{i:05d}" for i in range(len(df))]
    df["is_anomaly"] = (df["anomaly_type"] != "normal").astype(int)

    features, labels = df[FEATURE_COLS], df[LABEL_COLS]
    validate_features(features)
    return features, labels


def main():
    config = load_config()
    features, labels = generate_dataset(config)
    (ROOT / "data").mkdir(exist_ok=True)
    features.to_csv(ROOT / config["paths"]["features"], index=False)
    labels.to_csv(ROOT / config["paths"]["labels"], index=False)
    print(f"Saved {len(features)} accounts ({labels['is_anomaly'].mean():.1%} anomalous)")


if __name__ == "__main__":
    main()