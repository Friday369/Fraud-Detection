import yaml
import joblib
import pandas as pd
import time
import logging
from pydantic import BaseModel, ConfigDict, Field

from src.features import build_features, prepare_for_model
from src.explain import explain_account
from fastapi import FastAPI
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anomaly_api")
# 1. Read the config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

api_config = config["api"]

# 2. Load the scaler and model ONE time, when the service starts
scaler = joblib.load(api_config["scaler_path"])
model = joblib.load(api_config["model_path"])
score_cutoff = api_config["score_cutoff"]

# 3. Create the app
app = FastAPI(title="Account Anomaly Detection")


# 4. Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
class AccountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")   # reject unknown fields

    account_age_days: float = Field(ge=0)
    trades_per_day: float = Field(ge=0)
    baseline_trades_per_day: float = Field(ge=0)
    linked_account_count: int = Field(ge=0)
    timing_regularity: float = Field(ge=0)


@app.post("/score")
def score(account: AccountInput):
    start = time.time()
    # 1. One-row table with the raw columns
    raw = pd.DataFrame([account.model_dump()])

    # 2. Same feature code as training
    feats = build_features(raw)
    scaled = prepare_for_model(feats, scaler)

    # 3. Suspicion score (higher = more suspicious, same as Step 1)
    suspicion_score = float(-model.score_samples(scaled)[0])

    # 4. Flag against the saved cutoff
    flagged = suspicion_score > score_cutoff

    # 5. Plain-English reason
    explanation = explain_account(feats.iloc[0], scaled.iloc[0])
    ms = (time.time() - start) * 1000
    logger.info(f"score request: score={suspicion_score:.4f} flagged={flagged} time={ms:.1f}ms")


    return {
        "suspicion_score": round(suspicion_score, 4),
        "flagged": flagged,
        "explanation": explanation,
    }