# Streamlit review page (local demo)

A simple local Streamlit page for looking up one account and seeing its
suspicion score, flagged or not flagged, and a one-sentence reason.

## How it works

1. The analyst types an account ID (for example `ACC00267`) and presses
   Enter or clicks **Score**.
2. `src/lookup.py` finds that account's 5 raw fields in `data/features.csv`.
3. `app.py` sends those fields to `POST /score` on the API.
4. The page shows the score, the flagged status and the explanation.

## How to run

Start two processes in two separate terminals, from the project root,
with the virtual environment active.

Terminal 1 (the API):

    uvicorn src.api:app

Terminal 2 (the page):

    streamlit run app.py

The API address is set in `config.yaml` under `api.url`
(default `http://127.0.0.1:8000`).

## Messages the page can show

- Empty box: "Please type an account ID."
- Unknown ID: "Account '...' was not found."
- API not running: "Cannot reach the scoring API. Start it with: uvicorn src.api:app"
- Any non-200 reply from the API: "The API replied with an error (status ...)."

## Honest limits

- The page reads **simulated** accounts from `data/features.csv`. That file
  stands in for a platform's account database.
- It runs on my own machine only. It is not deployed and has not been used
  by real analysts.
- It scores one account at a time, using the saved score cutoff from the API.
  It does not apply the 3% batch budget.
- A flag is a reason for a human to review an account. Nothing happens
  automatically.
- The explanation shows which features looked unusual, not why. Accounts
  that are not flagged can still get a short or generic explanation.
- The "API replied with an error" message has not been triggered by hand.