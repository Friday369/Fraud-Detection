FIELDS = [
    "account_age_days",
    "trades_per_day",
    "baseline_trades_per_day",
    "linked_account_count",
    "timing_regularity",
]


def clean_id(account_id):
    """Remove spaces and make the ID upper case, so ' acc00267 ' works."""
    return str(account_id).strip().upper()


def get_account(account_id, table):
    """Return the account's 5 raw fields as a dict, or None if not found."""
    wanted_id = clean_id(account_id)

    matches = table[table["account_id"] == wanted_id]
    if matches.empty:
        return None

    row = matches.iloc[0]

    account = {}
    for field in FIELDS:
        account[field] = float(row[field])
    account["linked_account_count"] = int(row["linked_account_count"])
    return account