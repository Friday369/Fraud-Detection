# API Reference

A small FastAPI service that scores one trading account at a time and explains the result in plain English. It runs locally only. All data behind the model is simulated, so scores show that the pipeline works, not how it would perform on real fraud.

## Start the service

From the project root (the folder that contains `config.yaml`):

```
uvicorn src.api:app --reload
```

The service is then available at `http://127.0.0.1:8000`. FastAPI also provides an interactive test page at `http://127.0.0.1:8000/docs`.

At startup the service reads `config.yaml` and loads the saved scaler and Isolation Forest once. Every request reuses them.

## Configuration

The `api` section of `config.yaml` controls the service:

| Key | Meaning |
|---|---|
| `scaler_path` | Saved StandardScaler (`artifacts/scaler.pkl`) |
| `model_path` | Saved Isolation Forest (`artifacts/isolation_forest.pkl`) |
| `score_cutoff` | Score above which a single account is flagged |

## How the flag decision works

The Phase 5 batch experiment flags the **top 3%** of accounts. That budget only makes sense when many accounts are scored together, so a single account cannot be ranked against others.

For `/score`, the cutoff is taken from the training data: it is the 97th percentile of Isolation Forest scores on the training accounts (about 0.5603 in the current config). 3% of training accounts scored above it. An account is flagged when its score is above this saved cutoff.

Because the cutoff is fixed, the share of flagged accounts in new data can be higher or lower than 3%. Daily batch scoring of all accounts uses the true top 3% and is done by running the Phase 5 code.

## Endpoints

### GET /health

Checks that the service is alive.

**Response (200)**

```json
{"status": "ok"}
```

### POST /score

Scores one account from its raw fields.

**Request body (JSON)**

| Field | Type | Rule | Meaning |
|---|---|---|---|
| `account_age_days` | number | at least 0 | Days since signup |
| `trades_per_day` | number | at least 0 | Recent average trades per day |
| `baseline_trades_per_day` | number | at least 0 | The account's own long-run typical activity |
| `linked_account_count` | integer | at least 0 | Other accounts sharing its device or IP |
| `timing_regularity` | number | at least 0 | Variation in time gaps between trades (low means machine-like) |

All five fields are required. Unknown fields (for example `account_id`) are rejected.

**Response (200)**

| Field | Meaning |
|---|---|
| `suspicion_score` | Isolation Forest score, higher means more suspicious |
| `flagged` | `true` if the score is above the saved cutoff |
| `explanation` | One plain-English sentence built from the two most unusual features |

A zero `baseline_trades_per_day` is allowed. The spike ratio then defaults to 1.0 instead of dividing by zero.

## Example 1: a normal account

**Request**

```json
{
  "account_age_days": 213,
  "trades_per_day": 9.15,
  "baseline_trades_per_day": 12.96,
  "linked_account_count": 0,
  "timing_regularity": 0.874
}
```

**Response (200)**

```json
{
  "suspicion_score": 0.4168,
  "flagged": false,
  "explanation": "No feature looks clearly suspicious on its own; review the account's raw values."
}
```

## Example 2: an obvious trial abuser

**Request**

```json
{
  "account_age_days": 1,
  "trades_per_day": 400,
  "baseline_trades_per_day": 15,
  "linked_account_count": 0,
  "timing_regularity": 0.8
}
```

**Response (200)**

```json
{
  "suspicion_score": <<paste your real score>>,
  "flagged": true,
  "explanation": "<<paste your real explanation>>"
}
```

## Errors

Invalid input returns status **422** with a message that names the problem field. The model is never called.

**Missing field** (here `timing_regularity` was left out)

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "timing_regularity"],
      "msg": "Field required"
    }
  ]
}
```

**Extra field** (here `account_id` was sent): the `msg` is "Extra inputs are not permitted".

**Negative value** (here `trades_per_day` was -5): the `msg` is "Input should be greater than or equal to 0".

(The real responses also include an `input` entry that repeats what was sent. It is left out above for readability.)

## Call the API from Python

```python
import requests

account = {
    "account_age_days": 213,
    "trades_per_day": 9.15,
    "baseline_trades_per_day": 12.96,
    "linked_account_count": 0,
    "timing_regularity": 0.874,
}

response = requests.post("http://127.0.0.1:8000/score", json=account)
print(response.status_code)
print(response.json())
```

## Logging

Each scored request writes one log line with the score, the flag and the processing time, for example:

```
2026-11-02 21:50:46,466 INFO score request: score=0.4168 flagged=False time=62.0ms
```

Raw account values are not logged.

## Limitations

- The service runs on a local machine only. It is not containerized, not deployed and has not handled real traffic.
- Scores come from a model trained on simulated data.
- The fixed cutoff is an approximation of the 3% budget (see above).
- The explanation shows which features looked unusual, not the underlying cause.
- A flag is a reason for a human to review an account. The system never bans accounts automatically.