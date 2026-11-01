"""Detectors and the alert budget (Phase 5)."""

import pandas as pd
from sklearn.ensemble import IsolationForest

BUDGET = 0.03   # flag the top 3% of accounts


def zscore_scores(X):
    """Method A: biggest absolute z-score across the features.
    Higher = more suspicious."""
    return X.abs().max(axis=1)


def train_isolation_forest(X, seed):
    """Method B: train Isolation Forest. No contamination rate is given."""
    return IsolationForest(n_estimators=100, random_state=seed).fit(X)


def isolation_forest_scores(model, X):
    """Flip the sign so higher = more suspicious, like the z-score method."""
    return pd.Series(-model.score_samples(X), index=X.index)


def flag_top(scores, budget=BUDGET):
    """True for the top `budget` share of accounts by score."""
    n_flag = int(round(budget * len(scores)))
    top_accounts = scores.sort_values(ascending=False).head(n_flag).index
    return pd.Series(scores.index.isin(top_accounts), index=scores.index)