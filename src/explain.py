"""Plain-English explanations for flagged accounts (Phase 6).

Rank features with the SCALED values, write the sentence with the REAL values.
"""

from src.features import FEATURES, build_features, prepare_for_model

MIN_SCORE = 1.5   # a feature must be at least this many std devs in the suspicious direction
FALLBACK = "No feature looks clearly suspicious on its own; review the account's raw values."

# +1 = a high value is suspicious, -1 = a low value is suspicious
DIRECTION = {
    "account_age_days":     -1,
    "trades_per_day":       +1,
    "spike_ratio":          +1,
    "linked_account_count": +1,
    "timing_regularity":    -1,
}


def make_piece(col, value):
    """One sentence piece for one feature, using the account's real value."""
    if col == "account_age_days":
        days = int(round(value))
        if days <= 0:
            return "it was created today"
        word = "day" if days == 1 else "days"
        return f"it is only {days} {word} old"
    if col == "trades_per_day":
        return f"it makes {value:.0f} trades per day"
    if col == "spike_ratio":
        return f"its trading is {value:.1f}x its usual pace"
    if col == "linked_account_count":
        count = int(round(value))
        word = "account" if count == 1 else "accounts"
        return f"it shares a device or IP with {count} other {word}"
    if col == "timing_regularity":
        return "its trade timing is unusually regular"


def suspicion(scaled_row):
    """How suspicious each feature is for one account (bigger = more suspicious)."""
    return {col: scaled_row[col] * DIRECTION[col] for col in FEATURES}


def explain_account(feat_row, scaled_row):
    """One account: real feature values + scaled feature values -> one sentence."""
    score = suspicion(scaled_row)

    top_two = sorted(score, key=score.get, reverse=True)[:2]
    top_two = [col for col in top_two if score[col] >= MIN_SCORE]

    if len(top_two) == 0:
        return FALLBACK

    pieces = [make_piece(col, feat_row[col]) for col in top_two]
    sentence = " and ".join(pieces)
    return sentence[0].upper() + sentence[1:] + "."


def explain_raw(raw, scaler):
    """Raw account table (same columns as data/features.csv) -> list of explanations.
    This is the function the API will call: same build_features() as training."""
    feats = build_features(raw)
    scaled = prepare_for_model(feats, scaler)
    return [explain_account(feats.iloc[i], scaled.iloc[i]) for i in range(len(feats))]