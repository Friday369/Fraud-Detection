"""Tests for src/generate_data.py (Phase 2).

Run from the project root:
    pytest tests/test_data.py
"""

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src/` importable when pytest is run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import (  # noqa: E402
    FEATURE_COLS,
    LABEL_COLS,
    LABEL_ONLY_COLS,
    build_dataset,
    load_config,
    save_dataset,
    split_counts,
    validate,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def dataset(config):
    return build_dataset(config)


# ---------------------------------------------------------------------------
# Row counts and anomaly rate
# ---------------------------------------------------------------------------
def test_row_counts(config, dataset):
    features, labels = dataset
    n_normal = config["accounts"]["n_normal"]
    n_abuse = config["accounts"]["n_per_abuse_type"]
    expected = n_normal + 3 * n_abuse

    assert len(features) == expected
    assert len(labels) == expected


def test_anomaly_rate_is_about_three_percent(dataset):
    _, labels = dataset
    rate = labels["is_anomaly"].mean()
    assert 0.025 <= rate <= 0.035


def test_each_abuse_type_has_expected_count(config, dataset):
    _, labels = dataset
    n_abuse = config["accounts"]["n_per_abuse_type"]
    counts = labels["anomaly_type"].value_counts()

    for abuse_type in ["trial_abuse", "multi_account", "bot"]:
        assert counts[abuse_type] == n_abuse
    assert counts["normal"] == config["accounts"]["n_normal"]


def test_obvious_subtle_split(config, dataset):
    _, labels = dataset
    n_abuse = config["accounts"]["n_per_abuse_type"]
    frac = config["accounts"]["subtle_fraction"]
    n_obvious, n_subtle = split_counts(n_abuse, frac)

    for abuse_type in ["trial_abuse", "multi_account", "bot"]:
        subset = labels[labels["anomaly_type"] == abuse_type]
        assert (subset["difficulty"] == "obvious").sum() == n_obvious
        assert (subset["difficulty"] == "subtle").sum() == n_subtle


# ---------------------------------------------------------------------------
# No label leakage
# ---------------------------------------------------------------------------
def test_features_have_no_label_columns(dataset):
    features, _ = dataset
    assert list(features.columns) == FEATURE_COLS
    assert LABEL_ONLY_COLS.isdisjoint(features.columns)


def test_labels_have_only_label_columns(dataset):
    _, labels = dataset
    assert list(labels.columns) == LABEL_COLS


def test_saved_features_file_has_no_label_columns(config, dataset, tmp_path):
    features, labels = dataset
    features_path, labels_path = save_dataset(features, labels, config, root=tmp_path)

    saved_features = pd.read_csv(features_path)
    saved_labels = pd.read_csv(labels_path)

    assert LABEL_ONLY_COLS.isdisjoint(saved_features.columns)
    assert LABEL_ONLY_COLS <= set(saved_labels.columns)
    assert len(saved_features) == len(saved_labels)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validation_passes_on_generated_data(dataset):
    features, labels = dataset
    assert validate(features, labels) is True


def test_validation_catches_missing_values(dataset):
    features, labels = dataset
    bad = features.copy()
    bad.loc[0, "trades_per_day"] = None
    with pytest.raises(ValueError, match="Missing values"):
        validate(bad, labels)


def test_validation_catches_label_leakage(dataset):
    features, labels = dataset
    bad = features.copy()
    bad["is_anomaly"] = labels["is_anomaly"]
    with pytest.raises(ValueError):
        validate(bad, labels)


def test_validation_catches_out_of_range_values(dataset):
    features, labels = dataset
    bad = features.copy()
    bad.loc[0, "linked_account_count"] = 50
    with pytest.raises(ValueError, match="linked_account_count"):
        validate(bad, labels)


# ---------------------------------------------------------------------------
# Abuse patterns match the design
# ---------------------------------------------------------------------------
def _joined(dataset):
    features, labels = dataset
    return features.merge(labels, on="account_id")


def test_trial_abuse_accounts_are_new(dataset):
    df = _joined(dataset)
    trial = df[df["anomaly_type"] == "trial_abuse"]
    assert trial["account_age_days"].between(0, 3).all()


def test_trial_abuse_spike_ratio_ranges(dataset):
    df = _joined(dataset)
    df["spike"] = df["trades_per_day"] / df["baseline_trades_per_day"]
    trial = df[df["anomaly_type"] == "trial_abuse"]

    obvious = trial[trial["difficulty"] == "obvious"]["spike"]
    subtle = trial[trial["difficulty"] == "subtle"]["spike"]

    # small tolerance because trades_per_day and baseline are rounded
    assert obvious.between(9.5, 30.5).all()
    assert subtle.between(2.8, 5.2).all()


def test_multi_account_link_ranges(dataset):
    df = _joined(dataset)
    multi = df[df["anomaly_type"] == "multi_account"]

    obvious = multi[multi["difficulty"] == "obvious"]["linked_account_count"]
    subtle = multi[multi["difficulty"] == "subtle"]["linked_account_count"]

    assert obvious.between(6, 12).all()
    assert subtle.between(3, 4).all()


def test_bot_timing_ranges(dataset):
    df = _joined(dataset)
    bots = df[df["anomaly_type"] == "bot"]

    obvious = bots[bots["difficulty"] == "obvious"]["timing_regularity"]
    subtle = bots[bots["difficulty"] == "subtle"]["timing_regularity"]

    assert obvious.between(0.01, 0.10).all()
    assert subtle.between(0.30, 0.44).all()


def test_normal_accounts_stay_in_normal_range(dataset):
    df = _joined(dataset)
    normal = df[df["anomaly_type"] == "normal"]

    assert normal["account_age_days"].between(1, 900).all()
    assert normal["linked_account_count"].between(0, 2).all()
    assert normal["timing_regularity"].between(0.45, 2.0).all()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_data(config):
    f1, l1 = build_dataset(config)
    f2, l2 = build_dataset(config)
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(l1, l2)


def test_different_seed_gives_different_data(config):
    other = copy.deepcopy(config)
    other["seed"] = config["seed"] + 1

    f1, _ = build_dataset(config)
    f2, _ = build_dataset(other)
    assert not f1.equals(f2)


def test_ids_do_not_reveal_labels(dataset):
    """Shuffling happens before IDs are assigned, so anomalies are spread out."""
    _, labels = dataset
    first_half = labels.iloc[: len(labels) // 2]["is_anomaly"].sum()
    total = labels["is_anomaly"].sum()
    # anomalies should not all sit in one half of the ID range
    assert 0.2 * total < first_half < 0.8 * total