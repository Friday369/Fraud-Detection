from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pytest
from src import features
from src.features import build_features, fit_scaler, prepare_for_model, FEATURES


def sample_raw():
    return pd.DataFrame({
        "account_id": ["A1", "A2", "A3"],
        "account_age_days": [1, 300, 600],
        "trades_per_day": [400.0, 8.0, 5.0],
        "baseline_trades_per_day": [20.0, 8.0, 0.0],   # A3 has a zero baseline
        "linked_account_count": [1, 0, 9],
        "timing_regularity": [1.0, 0.9, 0.02],
    })


def test_output_columns():
    assert list(build_features(sample_raw()).columns) == FEATURES


def test_no_missing_or_infinite_values():
    out = build_features(sample_raw())
    assert out.isna().sum().sum() == 0
    assert not np.isinf(out.values).any()


def test_zero_baseline_is_handled():
    out = build_features(sample_raw())
    assert out.loc["A3", "spike_ratio"] == 1.0


def test_spike_ratio_value():
    out = build_features(sample_raw())
    assert out.loc["A1", "spike_ratio"] == 20.0


def test_missing_column_is_rejected():
    with pytest.raises(ValueError):
        build_features(sample_raw().drop(columns=["timing_regularity"]))


def test_scaler_reloads_identically(tmp_path):
    feats = build_features(sample_raw())
    scaler = fit_scaler(feats)
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    loaded = joblib.load(tmp_path / "scaler.pkl")
    assert np.allclose(prepare_for_model(feats, scaler),
                       prepare_for_model(feats, loaded))


def test_feature_code_does_not_touch_label_data():
    source = Path(features.__file__).read_text()
    assert "labels" not in source