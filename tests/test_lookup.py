import pandas as pd

from src.lookup import get_account


def make_table():
    """A tiny fake account table with two accounts."""
    return pd.DataFrame(
        {
            "account_id": ["ACC00001", "ACC00002"],
            "account_age_days": [100, 0],
            "trades_per_day": [8.0, 476.5],
            "baseline_trades_per_day": [7.5, 23.7],
            "linked_account_count": [1, 0],
            "timing_regularity": [30.0, 1.4],
        }
    )


def test_known_id_is_found():
    account = get_account("ACC00002", make_table())
    assert account["trades_per_day"] == 476.5
    assert account["account_age_days"] == 0


def test_unknown_id_returns_none():
    assert get_account("NOPE", make_table()) is None


def test_account_id_is_not_in_result():
    account = get_account("ACC00001", make_table())
    assert "account_id" not in account
    assert len(account) == 5


def test_linked_account_count_is_int():
    account = get_account("ACC00001", make_table())
    assert isinstance(account["linked_account_count"], int)


def test_spaces_and_lowercase_still_work():
    account = get_account("  acc00001 ", make_table())
    assert account is not None
    assert account["linked_account_count"] == 1