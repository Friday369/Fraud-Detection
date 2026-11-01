"""Tests for the detectors and evaluation (Phase 5)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import load_config, build_dataset
from features import build_features, fit_scaler, prepare_for_model
from models import (zscore_scores, train_isolation_forest,
                    isolation_forest_scores, flag_top)
from evaluate import run_experiment


@pytest.fixture(scope="module")
def data():
    """One small dataset, built once and shared by all tests."""
    config = load_config()
    config["seed"] = 7
    raw, labels = build_dataset(config)
    labels = labels.set_index("account_id")
    features = build_features(raw)
    X = prepare_for_model(features, fit_scaler(features))
    return X, labels


def test_scores_have_right_shape(data):
    X, _ = data
    z = zscore_scores(X)
    iso = isolation_forest_scores(train_isolation_forest(X, seed=1), X)
    assert len(z) == len(X)
    assert len(iso) == len(X)
    assert z.index.equals(X.index)
    assert iso.index.equals(X.index)


def test_exactly_three_percent_flagged(data):
    X, _ = data
    expected = int(round(0.03 * len(X)))
    z_flagged = flag_top(zscore_scores(X))
    iso_scores = isolation_forest_scores(train_isolation_forest(X, seed=1), X)
    iso_flagged = flag_top(iso_scores)
    assert z_flagged.sum() == expected
    assert iso_flagged.sum() == expected


def test_flagged_accounts_have_the_highest_scores(data):
    X, _ = data
    scores = zscore_scores(X)
    flagged = flag_top(scores)
    assert scores[flagged].min() >= scores[~flagged].max()


def test_same_seed_gives_identical_results(data):
    X, labels = data
    first = run_experiment(X, labels, seed=3)
    second = run_experiment(X, labels, seed=3)
    pd.testing.assert_frame_equal(first, second)