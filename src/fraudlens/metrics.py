
from prometheus_client import Counter, Histogram


API_REQUESTS = Counter(
    "fraudlens_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)


API_LATENCY = Histogram(
    "fraudlens_api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
    ),
)


MODEL_PREDICTIONS = Counter(
    "fraudlens_model_predictions_total",
    "Total successful model predictions",
    ["label"],
)


MODEL_FAILURES = Counter(
    "fraudlens_model_failures_total",
    "Total model inference failures",
)


MODEL_LATENCY = Histogram(
    "fraudlens_model_inference_duration_seconds",
    "Model inference duration in seconds",
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
    ),
)


RATE_LIMIT_REJECTIONS = Counter(
    "fraudlens_rate_limit_rejections_total",
    "Total requests rejected by rate limiter",
)


# Initialize prediction labels.
MODEL_PREDICTIONS.labels(label="Fraud")
MODEL_PREDICTIONS.labels(label="Legitimate")
