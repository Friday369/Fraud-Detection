# Feature Rationales

The model sees five features per account. All of them are built by `build_features()` in `src/features.py`. The same code runs in training and in the API.

Note: the data is simulated, so these rationales explain why each feature should help against the abuse patterns I generated, not against real fraud.

## The five features

| # | Feature | What it measures | Why it helps | Main abuse type |
|---|---|---|---|---|
| 1 | `account_age_days` | Days since the account was created | Trial abuse happens on brand-new accounts, so a very young age is a warning sign. On its own it is weak, because many normal accounts are also young. It matters in combination with high trading volume. | Trial abuse |
| 2 | `trades_per_day` | Recent average number of trades per day | Shows the raw activity level. Trial abusers trade far more than a typical account. | Trial abuse |
| 3 | `spike_ratio` | `trades_per_day / baseline_trades_per_day` | Compares an account to its own normal pace, so it catches a sudden jump even if the account is not a heavy trader by absolute numbers. A zero baseline is set to 1.0 (no known history, so no evidence of a spike). | Trial abuse |
| 4 | `linked_account_count` | Number of other accounts sharing the same device or IP | Most normal accounts have 0 or 1 linked accounts. Many linked accounts suggest one person or ring running several accounts. | Multi-accounting |
| 5 | `timing_regularity` | Variation in the time gaps between trades | Humans trade at irregular times. A value close to zero means machine-like, evenly spaced trading. | Bots |

## How the features are prepared before modelling

- **Log transform on `trades_per_day` and `spike_ratio`.** A few accounts trade hundreds of times per day, and without a log those values would dominate the scaling. The log keeps the order of accounts, so the extreme accounts are still the most extreme, just compressed. It worked well for trades per day (skew 19.48 to 0.69). For the spike ratio the skew only fell from 13.99 to 7.91, because a few real spikes remain far from the normal range. That is expected, since those accounts are the signal, and the log still compresses them.
- **StandardScaler on all five features.** Features have very different ranges (age up to about 900, linked accounts up to about 12). Scaling puts each on a comparable footing (mean 0, standard deviation 1), so no feature dominates just because of its units.
- **The scaler is fitted once and saved** to `artifacts/scaler.pkl`. The API loads the same scaler, so a new account is scaled exactly like the training data.
- **Labels are never used.** The feature code does not read `labels.csv`. A test checks this.

## What these features cannot do

- They describe a single snapshot of each account, not behaviour over time.
- A subtle abuser (for example, an account only 3 to 5 times above its normal pace) can look close to a busy normal account, so some will be missed.
- Each feature shows that something looks unusual, not why it happened.
