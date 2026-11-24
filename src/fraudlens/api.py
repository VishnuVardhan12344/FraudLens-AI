
from contextlib import asynccontextmanager
from pathlib import Path

import math
import os
import secrets
import time

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Security,
    Request,
)

from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from fraudlens.middleware import request_monitoring_middleware
from fraudlens.prediction_monitor import prediction_monitor
from fraudlens.rate_limiter import enforce_rate_limit
from fastapi import Response

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from fraudlens.metrics import (
    MODEL_PREDICTIONS,
    MODEL_FAILURES,
    MODEL_LATENCY,
)

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

    expected_api_key = os.getenv("FRAUDLENS_API_KEY")

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


# -------------------------------------
# MODEL CONFIGURATION
# -------------------------------------

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


# -------------------------------------
# TRANSACTION VALIDATION
# -------------------------------------

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

        if not all(
            math.isfinite(v) for v in values.values()
        ):
            raise ValueError(
                "All feature values must be finite"
            )

        if values["Time"] < 0:
            raise ValueError("Time cannot be negative")

        if values["Amount"] < 0:
            raise ValueError("Amount cannot be negative")

        return values


# -------------------------------------
# MODEL LOADING
# -------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    # Only load trusted model artifacts.
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


# -------------------------------------
# APPLICATION LIFESPAN
# -------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL_BUNDLE

    MODEL_BUNDLE = load_model()

    try:
        yield
    finally:
        MODEL_BUNDLE = None


# -------------------------------------
# FASTAPI APPLICATION
# -------------------------------------

app = FastAPI(
    title="FraudLens AI API",
    description="Experimental fraud prediction service",
    version="1.0.0",
    lifespan=lifespan,
)


# Request monitoring middleware
app.middleware("http")(request_monitoring_middleware)


# -------------------------------------
# HOME ENDPOINT
# -------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to FraudLens AI",
        "documentation": "/docs",
    }


# -------------------------------------
# HEALTH ENDPOINT
# -------------------------------------

@app.get("/health")
def health():

    return {
        "status": (
            "ready" if MODEL_BUNDLE is not None
            else "not_ready"
        )
    }


# -------------------------------------
# FEATURE ENGINEERING
# -------------------------------------

def prepare_features(raw_features, saved_features):

    row = dict(raw_features)

    amount = row["Amount"]
    elapsed_time = row["Time"]

    row["log_amount"] = math.log1p(amount)

    row["amount_is_zero"] = int(amount == 0)

    row["elapsed_day"] = int(
        elapsed_time // 86400
    )

    row["elapsed_hour"] = int(
        (elapsed_time % 86400) // 3600
    )

    frame = pd.DataFrame([row])

    return frame.loc[:, saved_features]


#
# -------------------------------------
# FRAUD PREDICTION ENDPOINT
# -------------------------------------

@app.post(
    "/predict",
    dependencies=[
        Depends(verify_api_key),
        Depends(enforce_rate_limit),
    ],
)
def predict(
    transaction: Transaction,
    request: Request,
):

    request_id = request.state.request_id

    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    model = MODEL_BUNDLE["model"]
    threshold = float(MODEL_BUNDLE["threshold"])
    feature_names = MODEL_BUNDLE["feature_names"]

    X = prepare_features(
        transaction.features,
        feature_names,
    )

    inference_start = time.perf_counter()

    try:

        probability = float(
            model.predict_proba(X)[0, 1]
        )

        inference_latency_ms = (
            time.perf_counter() - inference_start
        ) * 1000

    except Exception as exc:

        MODEL_FAILURES.inc()

        prediction_monitor.record_failure(
            request_id=request_id,
            error_type=type(exc).__name__,
        )

        raise

    if (
        not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):

        MODEL_FAILURES.inc()

        prediction_monitor.record_failure(
            request_id=request_id,
            error_type="InvalidProbability",
        )

        raise HTTPException(
            status_code=500,
            detail="Model returned invalid probability",
        )

    prediction = int(probability >= threshold)

    label = (
        "Fraud" if prediction == 1
        else "Legitimate"
    )

    prediction_monitor.record_prediction(
        probability=probability,
        prediction=prediction,
        latency_ms=inference_latency_ms,
        request_id=request_id,
    )

    # Prometheus model metrics
    MODEL_PREDICTIONS.labels(
        label=label
    ).inc()

    MODEL_LATENCY.observe(
        inference_latency_ms / 1000
    )

    return {
        "fraud_probability": round(probability, 6),
        "threshold": round(threshold, 4),
        "prediction": prediction,
        "label": label,
        "model": "RandomForest-Optimized",
    }


# -------------------------------------
# ML MONITORING ENDPOINT
# -------------------------------------

@app.get(
    "/monitoring/predictions",
    dependencies=[Depends(verify_api_key)],
)
def get_prediction_metrics():

    return prediction_monitor.get_metrics()


# -------------------------------------
# PROMETHEUS METRICS ENDPOINT
# -------------------------------------

@app.get(
    "/metrics",
    dependencies=[Depends(verify_api_key)],
)
def prometheus_metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )



# -------------------------------------
# ML MONITORING ENDPOINT
# -------------------------------------

@app.get(
    "/monitoring/predictions",
    dependencies=[Depends(verify_api_key)],
)
def get_prediction_metrics():

    return prediction_monitor.get_metrics()
