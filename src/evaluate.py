"""Grade both detectors with the hidden labels and run the 5-seed experiment (Phase 5).

Run from the project root:
    python src/evaluate.py
"""

from pathlib import Path

import joblib
import pandas as pd

from generate_data import load_config, build_dataset
from features import build_features, fit_scaler, prepare_for_model
from models import (zscore_scores, train_isolation_forest,
                    isolation_forest_scores, flag_top)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1, 2, 3, 4, 5]


def grade(flagged, labels):
    """Precision, recall, F1, and recall per type and difficulty."""
    is_abusive = labels["is_anomaly"].astype(bool)
    abusive = labels[is_abusive]

    caught = (flagged & is_abusive).sum()
    precision = caught / flagged.sum()
    recall = caught / is_abusive.sum()
    result = {"precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall)}

    flagged_abusive = flagged[abusive.index]
    for t in ["trial_abuse", "multi_account", "bot"]:
        result["recall_" + t] = flagged_abusive[abusive["anomaly_type"] == t].mean()
    for d in ["obvious", "subtle"]:
        result["recall_" + d] = flagged_abusive[abusive["difficulty"] == d].mean()
    return result


def run_experiment(X, labels, seed):
    """Score with both methods, flag the top 3%, grade. One row per method."""
    model = train_isolation_forest(X, seed)
    all_scores = {"zscore": zscore_scores(X),
                  "isolation_forest": isolation_forest_scores(model, X)}

    rows = []
    for method, scores in all_scores.items():
        row = {"seed": seed, "method": method}
        row.update(grade(flag_top(scores), labels))
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_seeds(seeds=SEEDS):
    """Full pipeline per seed: generate data, features, scale, score, grade."""
    results = []
    for seed in seeds:
        config = load_config()
        config["seed"] = seed
        raw, labels = build_dataset(config)
        labels = labels.set_index("account_id")

        features = build_features(raw)
        scaler = fit_scaler(features)
        X = prepare_for_model(features, scaler)

        results.append(run_experiment(X, labels, seed))
    return pd.concat(results, ignore_index=True)


def summarize(all_results):
    """Mean and std over seeds, one row per method."""
    grouped = all_results.drop(columns="seed").groupby("method")
    return grouped.mean().add_suffix("_mean").join(
        grouped.std().add_suffix("_std")).round(3)


def main():
    config = load_config()

    # 1) The 5-seed experiment -> results table
    summary = summarize(run_all_seeds())
    summary.to_csv(PROJECT_ROOT / "data" / "model_comparison.csv")

    # 2) Final Isolation Forest on the main dataset, saved for the API
    X = pd.read_csv(PROJECT_ROOT / "data" / "feature_matrix.csv", index_col="account_id")
    model = train_isolation_forest(X, config["seed"])
    joblib.dump(model, PROJECT_ROOT / "artifacts" / "isolation_forest.pkl")

    print(summary.T)
    print("\nSaved data/model_comparison.csv and artifacts/isolation_forest.pkl")


if __name__ == "__main__":
    main()