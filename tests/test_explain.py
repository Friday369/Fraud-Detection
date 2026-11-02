import os

import joblib
import pandas as pd
import pytest

from src.explain import FALLBACK, explain_raw, make_piece


@pytest.fixture(scope="module")
def scaler():
    return joblib.load("artifacts/scaler.pkl")


def one_account(age, trades, baseline, linked, timing):
    return pd.DataFrame([{
        "account_age_days": age,
        "trades_per_day": trades,
        "baseline_trades_per_day": baseline,
        "linked_account_count": linked,
        "timing_regularity": timing,
    }])


def test_every_account_gets_a_non_empty_explanation(scaler):
    raw = pd.read_csv("data/features.csv")
    explanations = explain_raw(raw, scaler)
    assert len(explanations) == len(raw)
    assert all(isinstance(e, str) and len(e) > 0 for e in explanations)


def test_saved_review_file_has_no_empty_explanations():
    path = "data/flagged_case_reviews.csv"
    if not os.path.exists(path):
        pytest.skip("review file not generated yet")
    reviews = pd.read_csv(path)
    assert reviews["explanation"].notna().all()
    assert (reviews["explanation"].str.len() > 0).all()


def test_quoted_values_match_the_input(scaler):
    # spike ratio = 500 / 10 = 50
    text = explain_raw(one_account(1, 500, 10, 0, 1.0), scaler)[0]
    assert "500 trades per day" in text
    assert "50.0x its usual pace" in text


def test_zero_baseline_gives_no_nonsense(scaler):
    text = explain_raw(one_account(1, 500, 0, 0, 1.0), scaler)[0]
    for bad in ["inf", "nan", "infinite"]:
        assert bad not in text.lower()
    assert "500 trades per day" in text


def test_grammar_for_one_and_zero():
    assert make_piece("account_age_days", 1) == "it is only 1 day old"
    assert make_piece("account_age_days", 0) == "it was created today"
    assert make_piece("linked_account_count", 1) == "it shares a device or IP with 1 other account"
    assert make_piece("linked_account_count", 9) == "it shares a device or IP with 9 other accounts"


def test_normal_account_gets_the_fallback(scaler):
    text = explain_raw(one_account(400, 8, 8, 0, 1.0), scaler)[0]
    assert text == FALLBACK


def test_missing_column_is_rejected(scaler):
    bad = one_account(1, 500, 10, 0, 1.0).drop(columns=["linked_account_count"])
    with pytest.raises(ValueError):
        explain_raw(bad, scaler)


def test_same_input_gives_same_output(scaler):
    row = one_account(1, 500, 10, 3, 0.02)
    assert explain_raw(row, scaler) == explain_raw(row, scaler)