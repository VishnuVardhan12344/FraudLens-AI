
from contextlib import asynccontextmanager
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import os
import secrets
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator



# -------------------------------------
# API KEY AUTHENTICATION
# -------------------------------------

load_dotenv()

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def verify_api_key(
    api_key: str | None = Security(api_key_header),
):
    expected_api_key = os.getenv(
        "FRAUDLENS_API_KEY"
    )

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not api_key or not secrets.compare_digest(
        api_key.encode("utf-8"),
        expected_api_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
        )

    return True



ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "random_forest_optimized.joblib"
)

FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

ENGINEERED_FEATURES = [
    "log_amount",
    "amount_is_zero",
    "elapsed_day",
    "elapsed_hour",
]

MODEL_BUNDLE = None


class Transaction(BaseModel):
    features: dict[str, float] = Field(
        description="Time, V1-V28, and Amount"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, values):

        required = set(FEATURE_NAMES)
        supplied = set(values)

        missing = sorted(required - supplied)
        extra = sorted(supplied - required)

        if missing or extra:
            raise ValueError(
                f"Missing features: {missing}; "
                f"Unexpected features: {extra}"
            )

        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(
                "All feature values must be finite"
            )

        if values["Time"] < 0:
            raise ValueError("Time cannot be negative")

        if values["Amount"] < 0:
            raise ValueError("Amount cannot be negative")

        return values


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    # Only load artifacts produced by trusted training code.
    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Expected Lesson 9 optimized model dictionary"
        )

    required_keys = {
        "model",
        "threshold",
        "feature_names",
    }

    if not required_keys.issubset(bundle):
        raise ValueError(
            "Model artifact is missing required keys"
        )

    model = bundle["model"]
    threshold = float(bundle["threshold"])
    saved_features = list(bundle["feature_names"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid model threshold")

    if (
        len(saved_features) != len(set(saved_features))
        or set(saved_features)
        != set(FEATURE_NAMES + ENGINEERED_FEATURES)
    ):
        raise ValueError(
            "Saved model features do not match API features"
        )

    if list(model.classes_) != [0, 1]:
        raise ValueError(
            "Expected binary model classes [0, 1]"
        )

    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE

    MODEL_BUNDLE = load_model()
    yield
    MODEL_BUNDLE = None


app = FastAPI(
    title="FraudLens AI API",
    description="Experimental fraud prediction service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "Welcome to FraudLens AI",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": (
            "ready" if MODEL_BUNDLE is not None
            else "not_ready"
        )
    }


def prepare_features(raw_features, saved_features):

    row = dict(raw_features)

    amount = row["Amount"]
    time = row["Time"]

    # Match Lesson 3 transformations.
    row["log_amount"] = math.log1p(amount)
    row["amount_is_zero"] = int(amount == 0)
    row["elapsed_day"] = int(time // 86400)
    row["elapsed_hour"] = int((time % 86400) // 3600)

    # Preserve the exact feature order used in training.
    frame = pd.DataFrame([row])

    return frame.loc[:, saved_features]


@app.post(
    "/predict",
    dependencies=[Depends(verify_api_key)],
)
def predict(transaction: Transaction):

    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    model = MODEL_BUNDLE["model"]
    threshold = float(MODEL_BUNDLE["threshold"])
    feature_names = MODEL_BUNDLE["feature_names"]

    X = prepare_features(
        transaction.features,
        feature_names,
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    if not np.isfinite(probability):
        raise HTTPException(
            status_code=500,
            detail="Model returned invalid probability"
        )

    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(probability, 6),
        "threshold": round(threshold, 4),
        "prediction": prediction,
        "label": (
            "Fraud" if prediction == 1
            else "Legitimate"
        ),
        "model": "RandomForest-Optimized",
    }

