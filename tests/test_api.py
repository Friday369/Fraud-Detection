from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# A normal account (the one we tried in the docs page)
NORMAL = {
    "account_age_days": 213,
    "trades_per_day": 9.15,
    "baseline_trades_per_day": 12.96,
    "linked_account_count": 0,
    "timing_regularity": 0.874,
}

# An obvious trial abuser: brand new account, huge trading
ABUSER = {
    "account_age_days": 1,
    "trades_per_day": 400,
    "baseline_trades_per_day": 15,
    "linked_account_count": 0,
    "timing_regularity": 0.8,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_normal_account_not_flagged():
    response = client.post("/score", json=NORMAL)
    assert response.status_code == 200
    body = response.json()
    assert body["flagged"] is False
    assert isinstance(body["suspicion_score"], float)
    assert len(body["explanation"]) > 0


def test_obvious_abuser_is_flagged():
    response = client.post("/score", json=ABUSER)
    assert response.status_code == 200
    body = response.json()
    assert body["flagged"] is True
    assert len(body["explanation"]) > 0


def test_end_to_end_explanation_matches_input():
    """Raw input all the way to the explanation: the text should quote the real values."""
    body = client.post("/score", json=ABUSER).json()
    assert "1 day old" in body["explanation"] or "400 trades" in body["explanation"]


def test_missing_field_gives_error():
    bad = dict(NORMAL)
    del bad["timing_regularity"]
    response = client.post("/score", json=bad)
    assert response.status_code == 422


def test_extra_field_gives_error():
    bad = dict(NORMAL)
    bad["account_id"] = "ACC00000"
    response = client.post("/score", json=bad)
    assert response.status_code == 422


def test_negative_value_gives_error():
    bad = dict(NORMAL)
    bad["trades_per_day"] = -5
    response = client.post("/score", json=bad)
    assert response.status_code == 422