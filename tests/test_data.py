import pytest

from src.generate_data import (FEATURE_COLS, LABEL_COLS, generate_dataset,
                               load_config, validate_features)


@pytest.fixture(scope="module")
def data():
    config = load_config()
    features, labels = generate_dataset(config)
    return config, features, labels


def test_row_counts(data):
    config, features, labels = data
    a = config["accounts"]
    expected = a["n_normal"] + 3 * a["n_per_abuse_type"]
    assert len(features) == expected
    assert len(labels) == expected


def test_anomaly_rate_is_about_3_percent(data):
    _, _, labels = data
    assert 0.025 <= labels["is_anomaly"].mean() <= 0.04


def test_each_abuse_type_present_with_both_difficulties(data):
    _, _, labels = data
    abusive = labels[labels["is_anomaly"] == 1]
    assert set(abusive["anomaly_type"]) == {"trial_abuse", "multi_account", "bot"}
    assert set(abusive["difficulty"]) == {"obvious", "subtle"}


def test_no_label_columns_in_features(data):
    _, features, _ = data
    leaked = {"is_anomaly", "anomaly_type", "difficulty"} & set(features.columns)
    assert not leaked
    assert list(features.columns) == FEATURE_COLS


def test_features_and_labels_share_ids(data):
    _, features, labels = data
    assert list(features["account_id"]) == list(labels["account_id"])
    assert list(labels.columns) == LABEL_COLS


def test_same_seed_gives_same_data(data):
    config, features, _ = data
    again, _ = generate_dataset(config)
    assert features.equals(again)


def test_different_seed_gives_different_data(data):
    config, features, _ = data
    other, _ = generate_dataset(config, seed=7)
    assert not features.equals(other)


def test_validation_catches_missing_values(data):
    _, features, _ = data
    bad = features.copy()
    bad.loc[0, "trades_per_day"] = None
    with pytest.raises(ValueError):
        validate_features(bad)