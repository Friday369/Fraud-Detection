# Trading Account Anomaly Detection

A system that flags suspicious trading accounts (bots, trial abusers and people running multiple accounts) and explains in plain English why each account was flagged. It is exposed through a tested API and a simple review page.

Real fraud data is not public, so the project generates **simulated data in which the abusive accounts are known**. This makes it possible to measure how well each detection method works.

> The data is simulated. Every flag is a signal for a human to review. The system never bans accounts automatically.

---

## The problem

Trading platforms face three common abuse patterns:

| Pattern | Description |
|---|---|
| **Trial abuse** | Brand-new accounts that make a huge number of trades to use up free limits |
| **Multi-accounting** | Several accounts sharing the same device or IP, often run by one person or a fraud ring |
| **Bot activity** | Accounts trading with machine-like, very regular timing |

A risk or support analyst can only review about 3% of accounts per day, so the system flags the **top 3% by suspicion score** instead of using a fixed threshold, and gives a short reason so each flag can be checked in seconds.

## What it does

- Scores an account from 0 to 1. The higher the score, the more unusual the account.
- Flags the account as yes or no.
- Gives a one-sentence reason, for example: *"Account is 1 day old and made 400 trades per day, 20x its usual pace."*
- Serves scores through a local API and a Streamlit review page where the analyst types only an account ID.

## The data

About 3,000 normal accounts and about 100 abusive accounts (roughly 3%), generated from a seed in `config.yaml` so the exact dataset can be reproduced.

| Column | Meaning |
|---|---|
| `account_id` | Unique ID |
| `account_age_days` | Days since signup |
| `trades_per_day` | Recent average trades per day |
| `baseline_trades_per_day` | The account's own long-run typical activity |
| `linked_account_count` | Other accounts sharing its device or IP |
| `timing_regularity` | Variation in the time gaps between trades (low means machine-like) |

The labels (`is_anomaly`, `anomaly_type`, `difficulty`) live in a **separate file** and are never seen by the models. They are used only to evaluate results.

About 33 accounts of each abuse type are created, half obvious and half subtle. The subtle half overlaps with normal behavior, so detectors can miss some and the results are meaningful.

| Type | Obvious version | Subtle version |
|---|---|---|
| Trial abuse | Age 0-3 days, trades 10-30x normal | Age 0-3 days, trades only 3-5x normal |
| Multi-account | 6-12 linked accounts | 3-4 linked accounts |
| Bot | Timing variation near zero | Timing just below the normal range |

## How it works

The project is built in 9 phases.

| Phase | What was done |
|---|---|
| 1. Requirements | Defined the users, inputs and outputs, the 3% alert budget and the success metrics. |
| 2. Simulated data | Built a config-driven generator with validation and a unit test. Features and labels are saved separately. |
| 3. Exploratory analysis | Compared normal and abusive accounts and found where subtle anomalies hide. Trades per day is heavily right-skewed (median 8.03, mean 12.87, max 879.69), with the extremes coming from trial-abuse accounts. |
| 4. Feature pipeline | One reusable `build_features()` function used in both training and serving, with a saved scaler and a saved feature list. |
| 5. Models and evaluation | Compared a z-score baseline against Isolation Forest at the 3% alert budget, per abuse type and obvious vs. subtle, over 5 random seeds. |
| 6. Explanations | Rule-based sentence templates that describe the top two most unusual features of each flagged account. |
| 7. API and tests | A FastAPI service with startup model loading, input validation, logging and tests. |
| 8. Review page | A Streamlit page that looks up an account by ID and calls the API. |
| 9. Wrap-up | Final README, docs, clean-up and final check. |

### The 5 features

| # | Feature | How it is computed | Why it helps |
|---|---|---|---|
| 1 | Account age | `account_age_days` | New accounts are the main trial-abuse signal |
| 2 | Trades per day | `trades_per_day` | Raw activity level |
| 3 | Order spike ratio | `trades_per_day / baseline_trades_per_day` | Catches sudden jumps compared to the account's own normal (zero baseline is handled) |
| 4 | Linked accounts | `linked_account_count` | Signals multi-accounting |
| 5 | Timing regularity | `timing_regularity` | Very low values suggest bots |

Heavy-tailed features are log-transformed (`log1p` cut the skew of trades per day from 19.48 to 0.69 and of spike ratio from 13.99 to 7.91) and then scaled with a StandardScaler. The fitted scaler is saved and reused by the API.

### The two detectors

- **Method A: z-score baseline.** For each account, take the largest z-score across features as its suspicion score. Simple and easy to explain, but looks at one feature at a time.
- **Method B: Isolation Forest.** Builds many random trees. Unusual accounts are isolated in fewer splits and get higher scores, so it can pick up combinations of features that look odd together.

Both methods flag the top 3% of accounts. The true anomaly rate is never given to the models. The whole experiment is repeated with 5 random seeds and reported as mean and spread. Results are written to `data/model_comparison.csv`.

### Flag cutoff for single accounts

The 3% budget only makes sense across many accounts at once. For `/score`, an account is flagged when its score is above **0.5603**, the 97th percentile of the Isolation Forest training scores (about 3% of training accounts are above it). The cutoff is stored in `config.yaml`.

### Design rules

- Unsupervised detection. Labels are used only for evaluation.
- No label leakage. The feature code never reads the labels file.
- Same feature code in training and serving.
- Explanations are rule-based templates, so they are deterministic and cannot invent facts.
- Config-driven, fixed seeds and pinned dependencies.
- Notebooks hold the full step-by-step code, and `src/` is the packaged copy used by the API and tests.

## Tech stack

Python 3.10+, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter, PyYAML, FastAPI, pytest, Streamlit, requests.

## Project structure

```
.
├── config.yaml                 # seed, counts, paths, score cutoff, API URL
├── requirements.txt            # pinned versions
├── app.py                      # Streamlit review page
├── data/
│   ├── features.csv            # raw simulated features
│   ├── labels.csv              # hidden labels, evaluation only
│   ├── feature_matrix.csv      # engineered, scaled features
│   ├── model_comparison.csv    # results table
│   └── flagged_case_reviews.csv  # flagged accounts with explanations
├── artifacts/
│   ├── scaler.pkl              # fitted StandardScaler
│   └── isolation_forest.pkl    # trained model
├── src/
│   ├── generate_data.py
│   ├── features.py             # build_features()
│   ├── models.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── api.py
│   └── lookup.py               # get_account() for the review page
├── tests/
├── notebooks/                  # full step-by-step code for each phase
└── docs/                       # requirements, data assumptions, API, figures
```

## Setup

```powershell
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

**1. Start the API**

```powershell
uvicorn src.api:app
```

**2. Start the review page (in a second terminal)**

```powershell
streamlit run app.py
```

Type an account ID, press Enter or click Score, and the page shows the score, whether the account is flagged, and the reason. The page has no scoring logic of its own. It calls the same API, so features, model and explanations stay in one place.

## API

| Endpoint | What it does |
|---|---|
| `POST /score` | Takes one account's raw fields and returns `suspicion_score`, `flagged` and `explanation`. |
| `GET /health` | Checks that the API is running. |

Bad input (missing or extra fields, values out of range) returns a clear error message instead of a crash. Full details are in [`docs/api.md`](docs/api.md).

## Run the tests

```powershell
pytest
```

## Example

| Account | Score | Flagged |
|---|---|---|
| ACC00000 | 0.4168 | No |
| ACC00267 | 0.7131 | Yes |
