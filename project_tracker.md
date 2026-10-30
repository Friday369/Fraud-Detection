# Project Tracker: Trading Account Anomaly Detection

A production-style service that scores trading accounts for abuse (bots, trial abuse, multi-accounting) and explains every flag in plain English.
Tick a box (`[x]`) when a task is done. Update the "Status" line at the top of each phase.

**Workflow for every phase:** write and understand the full working code in that phase's notebook first (my main material for revising), then copy the same logic into `src/` for the tests and the API. Keep the notebook and `src/` in sync.

**Overall progress:** 0 / 8 phases complete
**Scope:** core = Phases 1-8 (ends with the local API and wrap-up). Stretch items are parked at the bottom; do not start them until the core is done.

---

## Phase 1: Requirements and Success Criteria
**Status:** Not started
**Goal:** Decide what the system must do, for whom, and how success is measured, before writing code.

- [x] Write the problem statement (3 abuse types and why they matter to a trading platform)
- [x] Define the user (risk / support analyst) and the human-in-the-loop rule (no automatic bans)
- [x] Define the input (one row of account features) and output (score, flagged yes/no, one-sentence reason)
- [x] Define scoring modes: daily batch scoring and on-demand single-account API
- [x] Set the alert budget (flag the top 3% of accounts, not a fixed threshold)
- [x] Set success metrics (precision and recall at the 3% budget, per abuse type, averaged over 5 runs)
- [x] List non-goals (no auto-bans, no real-time streaming, no cloud deployment)
- [x] Write `docs/requirements.md` (one page)
- [x] Create the folder structure: `data/`, `src/`, `tests/`, `notebooks/`, `docs/`, `artifacts/`
- [x] Initialize the Git repo and make the first commit

**Can I explain:** who uses the system, why the top 3% instead of a fixed cutoff, and why a human stays in the loop?  [ ]

---

## Phase 2: Simulated Data
**Status:** Not started
**Goal:** Generate realistic trading accounts where I secretly know which ones are abusive.

- [x] Create `config.yaml` (random seed, account counts, fraction of subtle anomalies)
- [x] Write `src/generate_data.py`
- [x] Generate about 3,000 normal accounts (age, trades per day, baseline, linked accounts, timing regularity)
- [x] Generate about 33 trial-abuse accounts (half obvious, half subtle)
- [x] Generate about 33 multi-account cases (half obvious, half subtle)
- [x] Generate about 33 bot accounts (half obvious, half subtle)
- [x] Save features to `data/features.csv`
- [x] Save labels separately to `data/labels.csv` (`is_anomaly`, `anomaly_type`, `difficulty`)
- [x] Add a validation step (column types, value ranges, no missing values)
- [x] Write a unit test (row counts, about 3% anomaly rate, no label columns in the features file)
- [x] Write `docs/data_assumptions.md` stating clearly that the data is simulated and how each pattern was created
- [x] Build `notebooks/02_data_generation.ipynb` (full generator code step by step, showing how each account type is created, with sample rows and a chart)

**Can I explain:** why the data is simulated, how each abuse type was created, why half the anomalies are subtle, and why labels are in a separate file?  [ ]

---

## Phase 3: Exploratory Data Analysis
**Status:** Not started
**Goal:** Understand how abusive accounts differ from normal ones and where they hide.

- [x] Run the health check (shape, data types, missing values, summary statistics)
- [x] Chart: trades per day, normal vs. abusive (log scale)
- [x] Chart: linked account count by account type
- [x] Chart: timing regularity, normal vs. bot (histogram)
- [x] Chart: account age vs. trades per day (scatter, colored by type)
- [x] Plot obvious vs. subtle anomalies separately to find the overlap with normal accounts
- [x] Make a correlation heatmap of the features
- [x] Write a one-sentence takeaway under every chart
- [x] Save all figures to `docs/figures/`
- [x] Confirm the notebook runs top to bottom from a fresh start (`notebooks/03_eda.ipynb`)
- [x] Write a short "Findings" section (5 to 6 bullet points)

**Can I explain:** which features separate each abuse type, where subtle anomalies overlap with normal accounts, and what this means for the features I build next?  [ ]

---

## Phase 4: Feature Pipeline
**Status:** Not started
**Goal:** Turn raw columns into five model-ready features with one reusable function used in both training and serving.

- [ ] Write `build_features()` in `src/features.py`
- [ ] Feature 1: account age
- [ ] Feature 2: trades per day
- [ ] Feature 3: order spike ratio (handle the zero-baseline case)
- [ ] Feature 4: linked account count
- [ ] Feature 5: timing regularity
- [ ] Decide on a log transform for heavy-tailed columns (based on Phase 3)
- [ ] Fit StandardScaler and save it to `artifacts/scaler.pkl`
- [ ] Save the final feature list (so the API can reject bad inputs)
- [ ] Save `data/feature_matrix.csv`
- [ ] Write `docs/feature_rationales.md` (one line per feature, in my own words)
- [ ] Write `tests/test_features.py` (output columns, no missing values, zero-baseline case, scaler reloads identically)
- [ ] Confirm the feature code never touches `labels.csv`
- [ ] Build `notebooks/04_features.ipynb` (each feature built on a few example accounts, before and after scaling)

**Can I explain:** what each feature measures, what scaling is, why the scaler is saved, and why the same feature code runs in training and serving?  [ ]

---

## Phase 5: Models and Evaluation
**Status:** Not started
**Goal:** Build two detectors, compare them honestly, and show the results are not luck.

- [ ] Build Method A: z-score baseline (largest absolute z-score per account) in `src/models.py`
- [ ] Build Method B: Isolation Forest in `src/models.py`
- [ ] Score every account with both methods
- [ ] Apply the 3% alert budget to both (no peeking at the true anomaly rate)
- [ ] Compute precision, recall, and F1 for each method in `src/evaluate.py`
- [ ] Compute recall per abuse type (trial abuse, multi-account, bot)
- [ ] Compute recall for obvious vs. subtle anomalies
- [ ] (Optional) Plot precision-recall curves
- [ ] Repeat the full experiment with 5 random seeds and report mean and spread
- [ ] Save the Isolation Forest to `artifacts/isolation_forest.pkl`
- [ ] Save results to `data/model_comparison.csv`
- [ ] Make 1 or 2 result charts
- [ ] Write a plain-English conclusion (report what the results actually show)
- [ ] Write `tests/test_models.py` (score shape, exactly 3% flagged, same seed gives same results)
- [ ] Add a single command that runs the full experiment
- [ ] Build `notebooks/05_models_evaluation.ipynb` (score both methods, show flagged accounts, plot results)

**Can I explain:** how a z-score works, how Isolation Forest works intuitively, why the alert budget is fairer than a contamination rate, precision vs. recall, and why I ran 5 seeds?  [ ]

---

## Phase 6: Explanations
**Status:** Not started
**Goal:** Give every flagged account a short plain-English reason.

- [ ] Write sentence templates for all five features in `src/explain.py`
- [ ] Build the explanation function (top two most unusual features per account)
- [ ] Handle edge cases (zero baseline must not produce "infinite pace")
- [ ] Generate `data/flagged_case_reviews.csv` (account ID, score, method, explanation)
- [ ] Spot-check 10 to 15 explanations by hand against the raw data
- [ ] Measure explanation quality against the labels (does the explanation point at the manipulated feature?)
- [ ] Write `tests/test_explain.py` (non-empty explanations, quoted values match input, edge cases)
- [ ] Add a short doc note on how explanation quality was checked
- [ ] Build `notebooks/06_explanations.ipynb` (flagged accounts shown next to their explanations)

**Can I explain:** why explanations matter to reviewers, how the top two features are chosen, why templates are used instead of free-form text, and what the limits are?  [ ]

---

## Phase 7: API and Tests
**Status:** Not started
**Goal:** Serve the system through a small, tested API that runs on my machine.

**API (FastAPI)**
- [ ] Write `src/api.py` with `POST /score` (returns score, flagged, explanation)
- [ ] Add `GET /health`
- [ ] Load the scaler and model once at startup
- [ ] Run every request through `build_features()`, the model, and the explanation function
- [ ] Add input validation with clear error messages
- [ ] Save a score cutoff from training data for single-account flagging and document it
- [ ] Add basic logging (request time and result, no sensitive data)
- [ ] Move paths and the cutoff into `config.yaml`
- [ ] Write `docs/api.md` with example requests and responses

**Tests**
- [ ] Write `tests/test_api.py` (valid input, missing fields, a known obvious abuser is flagged)
- [ ] Add an end-to-end test (raw input to explanation)
- [ ] Confirm the whole test suite passes together
- [ ] Start the API locally and send a real request to it

**Can I explain:** what an API is and why it beats a notebook, why models load once at startup, why the single-account cutoff differs from the 3% batch budget, and how input validation protects the service?  [ ]

---

## Phase 8: Wrap-up (README, Resume, Interview Prep)
**Status:** Not started
**Goal:** Make the project easy for a recruiter to understand and easy for me to talk about.

**README and cleanup**
- [ ] Finish the README (problem, users, how it works, results table from code, how to run, API example, limitations)
- [ ] Add a "what I would do next" section (stretch ideas, described honestly as not built)
- [ ] Remove clutter (scratch notebooks, notes to self, unused files)
- [ ] Confirm a fresh clone runs from the README alone
- [ ] Tidy commit history and folder names

**Resume and interview prep**
- [ ] Write a 2 to 3 line resume entry with my real numbers
- [ ] Write a 30-second spoken pitch
- [ ] Write a 2-minute walkthrough
- [ ] Prepare answers to likely questions, especially "Why trust results on simulated data?"

**Can I explain:** the whole project in 30 seconds, and what it does not do and why I say so openly?  [ ]

---

## Core Final Checklist

- [ ] Phases 1-8 all ticked off
- [ ] All tests pass locally
- [ ] README states the data is simulated and lists limitations
- [ ] No claims on the resume that the project does not back up
- [ ] I can explain every part of the project without notes

---

# STRETCH (PARKED): do not start until the core is done

**Stretch progress:** 0 / 5. Suggested order: S1 to S5. Add each only once I can explain the previous part. Add a resume claim only after its item is built.

## S1: Docker
- [ ] Write the `Dockerfile`
- [ ] Pin library versions in `requirements.txt`
- [ ] `docker build` and `docker run` work from a clean state
- [ ] Add run instructions to the README

**Can I explain:** what Docker solves and why pinned versions matter?  [ ]

## S2: CI (GitHub Actions)
- [ ] Add `.github/workflows/tests.yml` to run tests on every push
- [ ] Add the pass/fail badge to the README

**Can I explain:** what CI does and why it catches broken changes early?  [ ]

## S3: Batch Endpoint
- [ ] Add `POST /score-batch` (top 3% flagged across the batch)
- [ ] Add tests for the batch endpoint
- [ ] Update `docs/api.md`
- [ ] (Optional) Build `notebooks/07_api_demo.ipynb` (send sample requests to the running API and show the responses)

**Can I explain:** why batch flagging uses the 3% budget while single scoring uses a fixed cutoff?  [ ]

## S4: Drift Monitoring
- [ ] Save a reference summary of the training features (`artifacts/reference_stats.json`)
- [ ] Write `src/monitor.py` (compare a new batch to the reference using PSI or the KS test)
- [ ] Output a drift report (which features drifted, by how much, "OK" or "Review model")
- [ ] Track the share of accounts flagged (alert if far from 3%)
- [ ] Create a shifted batch and show the monitor catching it
- [ ] Write `tests/test_monitor.py` (no drift on matching data, drift detected on the shifted batch)
- [ ] Make the monitor runnable from one command
- [ ] (Optional) Build `notebooks/08_monitoring_demo.ipynb`

**Can I explain:** what drift is, why models degrade, and how the monitor decides something changed?  [ ]

## S5: Model Card and Runbook
- [ ] Write `docs/model_card.md` (purpose, simulated training data, evaluation, weaknesses, what not to use it for)
- [ ] Write `docs/runbook.md` (what to do on "Review model", how to retrain, how to roll back; needs S4)

**Can I explain:** what a model card is for and what each runbook step means?  [ ]
