"""Simulated trading-account data generator (Phase 2).

Creates normal accounts plus three abuse types (trial abuse, multi-account,
bot), each split into an obvious and a subtle version. Writes two files:

  - features.csv : the columns the models are allowed to see
  - labels.csv   : the hidden truth, used only for evaluation

Run from the project root:
    python src/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

FEATURE_COLS = [
    "account_id",
    "account_age_days",
    "trades_per_day",
    "baseline_trades_per_day",
    "linked_account_count",
    "timing_regularity",
]
LABEL_COLS = ["account_id", "is_anomaly", "anomaly_type", "difficulty"]
LABEL_ONLY_COLS = {"is_anomaly", "anomaly_type", "difficulty"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path=DEFAULT_CONFIG_PATH):
    """Read the YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def split_counts(n, subtle_fraction):
    """Split n accounts into (n_obvious, n_subtle)."""
    n_subtle = int(round(n * subtle_fraction))
    return n - n_subtle, n_subtle


def generate_normal_accounts(n, rng):
    """Create n normal (non-abusive) trading accounts."""
    # Age: anywhere from 1 to 900 days since signup
    account_age_days = rng.integers(1, 901, size=n)

    # Baseline: each account's own typical daily trades.
    # Lognormal gives a median of ~8 with a long tail of heavy traders.
    baseline = rng.lognormal(mean=np.log(8), sigma=0.8, size=n)

    # Recent trades/day = baseline * small random wobble (~20%).
    trades_per_day = baseline * rng.lognormal(mean=0, sigma=0.2, size=n)

    # Linked accounts: mostly 0, sometimes 1, rarely 2
    linked_account_count = rng.choice([0, 1, 2], size=n, p=[0.75, 0.22, 0.03])

    # Timing regularity: humans are irregular, so values sit around 1.0,
    # with a floor of 0.45.
    timing_regularity = np.clip(rng.normal(loc=1.0, scale=0.25, size=n), 0.45, 2.0)

    return pd.DataFrame(
        {
            "account_age_days": account_age_days,
            "trades_per_day": trades_per_day.round(2),
            "baseline_trades_per_day": baseline.round(2),
            "linked_account_count": linked_account_count,
            "timing_regularity": timing_regularity.round(3),
        }
    )


def generate_trial_abuse(n, subtle_fraction, rng):
    """Brand-new accounts trading far above their own baseline."""
    n_obvious, n_subtle = split_counts(n, subtle_fraction)
    df = generate_normal_accounts(n, rng)

    # Multiplier on the account's baseline: obvious 10-30x, subtle 3-5x
    multiplier = np.concatenate(
        [
            rng.uniform(10, 30, size=n_obvious),
            rng.uniform(3, 5, size=n_subtle),
        ]
    )

    df["account_age_days"] = rng.integers(0, 4, size=n)  # 0-3 days old
    df["trades_per_day"] = (df["baseline_trades_per_day"] * multiplier).round(2)
    df["difficulty"] = ["obvious"] * n_obvious + ["subtle"] * n_subtle
    df["anomaly_type"] = "trial_abuse"
    return df


def generate_multi_account(n, subtle_fraction, rng):
    """Accounts sharing a device/IP with many other accounts."""
    n_obvious, n_subtle = split_counts(n, subtle_fraction)
    df = generate_normal_accounts(n, rng)

    # Obvious: 6-12 linked accounts. Subtle: 3-4 linked accounts.
    df["linked_account_count"] = np.concatenate(
        [
            rng.integers(6, 13, size=n_obvious),  # upper bound exclusive
            rng.integers(3, 5, size=n_subtle),
        ]
    )
    df["difficulty"] = ["obvious"] * n_obvious + ["subtle"] * n_subtle
    df["anomaly_type"] = "multi_account"
    return df


def generate_bots(n, subtle_fraction, rng):
    """Accounts trading with machine-like, very regular timing."""
    n_obvious, n_subtle = split_counts(n, subtle_fraction)
    df = generate_normal_accounts(n, rng)

    # Obvious: timing near zero. Subtle: just below the normal floor of 0.45.
    df["timing_regularity"] = np.concatenate(
        [
            rng.uniform(0.01, 0.10, size=n_obvious),
            rng.uniform(0.30, 0.44, size=n_subtle),
        ]
    ).round(3)
    df["difficulty"] = ["obvious"] * n_obvious + ["subtle"] * n_subtle
    df["anomaly_type"] = "bot"
    return df


# ---------------------------------------------------------------------------
# Build, validate, save
# ---------------------------------------------------------------------------
def build_dataset(config):
    """Generate all accounts and return (features, labels).

    The same seed always gives the same dataset.
    """
    rng = np.random.default_rng(config["seed"])  # one seeded generator
    n_normal = config["accounts"]["n_normal"]
    n_abuse = config["accounts"]["n_per_abuse_type"]
    frac = config["accounts"]["subtle_fraction"]

    # The order of these calls matters: it fixes how the random stream is used.
    normal = generate_normal_accounts(n_normal, rng)
    trial = generate_trial_abuse(n_abuse, frac, rng)
    multi = generate_multi_account(n_abuse, frac, rng)
    bots = generate_bots(n_abuse, frac, rng)

    # Normal accounts get label columns too, so everything stacks into one table
    normal["anomaly_type"] = "normal"
    normal["difficulty"] = "none"

    all_accounts = pd.concat([normal, trial, multi, bots], ignore_index=True)

    # Shuffle first, then assign IDs, so IDs carry no information about labels
    all_accounts = all_accounts.sample(frac=1, random_state=config["seed"]).reset_index(
        drop=True
    )
    all_accounts["account_id"] = [f"ACC{i:05d}" for i in range(len(all_accounts))]
    all_accounts["is_anomaly"] = (all_accounts["anomaly_type"] != "normal").astype(int)

    features = all_accounts[FEATURE_COLS].copy()
    labels = all_accounts[LABEL_COLS].copy()
    return features, labels


def validate(features, labels):
    """Check types, ranges, missing values and label leakage.

    Raises ValueError with a clear message if anything is wrong.
    """
    if list(features.columns) != FEATURE_COLS:
        raise ValueError(f"Unexpected feature columns: {list(features.columns)}")
    if list(labels.columns) != LABEL_COLS:
        raise ValueError(f"Unexpected label columns: {list(labels.columns)}")

    if not LABEL_ONLY_COLS.isdisjoint(features.columns):
        raise ValueError("Label column leaked into the features file")

    if features.isna().any().any():
        raise ValueError("Missing values found in features")
    if labels.isna().any().any():
        raise ValueError("Missing values found in labels")

    if not features["account_id"].is_unique:
        raise ValueError("account_id values are not unique")
    if not (features["account_id"] == labels["account_id"]).all():
        raise ValueError("features and labels account_id do not line up")

    numeric_cols = [c for c in FEATURE_COLS if c != "account_id"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(features[col]):
            raise ValueError(f"Column {col} is not numeric")

    if not (features["account_age_days"] >= 0).all():
        raise ValueError("account_age_days must be >= 0")
    if not (features["trades_per_day"] > 0).all():
        raise ValueError("trades_per_day must be > 0")
    if not (features["baseline_trades_per_day"] > 0).all():
        raise ValueError("baseline_trades_per_day must be > 0")
    if not features["linked_account_count"].between(0, 12).all():
        raise ValueError("linked_account_count must be between 0 and 12")
    if not (features["timing_regularity"] > 0).all():
        raise ValueError("timing_regularity must be > 0")

    if not set(labels["is_anomaly"].unique()) <= {0, 1}:
        raise ValueError("is_anomaly must be 0 or 1")
    return True


def save_dataset(features, labels, config, root=PROJECT_ROOT):
    """Write features.csv and labels.csv to the paths named in the config."""
    features_path = Path(root) / config["paths"]["features"]
    labels_path = Path(root) / config["paths"]["labels"]
    features_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)
    return features_path, labels_path


def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    features, labels = build_dataset(config)
    validate(features, labels)
    features_path, labels_path = save_dataset(features, labels, config)

    print(f"Features: {features.shape} -> {features_path}")
    print(f"Labels:   {labels.shape} -> {labels_path}")
    print(f"Anomaly rate: {labels['is_anomaly'].mean():.4f}")
    print(labels.groupby(["anomaly_type", "difficulty"]).size().to_string())


if __name__ == "__main__":
    main()