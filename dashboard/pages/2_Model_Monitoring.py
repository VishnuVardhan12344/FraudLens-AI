
import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="FraudLens Monitoring",
    page_icon="📊",
    layout="wide",
)

API_URL = os.getenv(
    "FRAUDLENS_API_URL",
    "https://fraudlens-ai-ecn5.onrender.com",
).rstrip("/")

API_KEY = os.getenv("FRAUDLENS_API_KEY")


st.title("FraudLens AI — Model Monitoring")

st.caption(
    "Monitor model predictions, inference latency "
    "and API availability."
)


if not API_KEY:
    st.error("API authentication is not configured.")
    st.stop()


headers = {
    "X-API-Key": API_KEY
}


@st.cache_data(ttl=15, show_spinner=False)
def fetch_metrics():

    with httpx.Client(timeout=120) as client:

        health = client.get(
            f"{API_URL}/health"
        )

        health.raise_for_status()

        response = client.get(
            f"{API_URL}/monitoring/predictions",
            headers=headers,
        )

        response.raise_for_status()

        return {
            "health": health.json(),
            "metrics": response.json(),
            "updated_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }


st.sidebar.header("Monitoring Configuration")

st.sidebar.write("Refresh cache: 15 seconds")

if st.sidebar.button("Refresh Metrics"):
    fetch_metrics.clear()
    st.rerun()


try:

    with st.spinner("Fetching monitoring metrics..."):
        data = fetch_metrics()

except httpx.HTTPStatusError as exc:

    st.error(
        f"API returned HTTP "
        f"{exc.response.status_code}"
    )

    st.stop()

except httpx.RequestError:

    st.error("Unable to connect to FraudLens API.")
    st.stop()


metrics = data["metrics"]

health = data["health"]["status"]


if health == "ready":
    st.success("API Status: Operational")
else:
    st.warning("API Status: Not Ready")


st.caption(
    f"Last updated: {data['updated_at']}"
)


# -------------------------------------
# KEY PERFORMANCE INDICATORS
# -------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Predictions",
    metrics["total_predictions"],
)

col2.metric(
    "Fraud Predictions",
    metrics["fraud_predictions"],
)

col3.metric(
    "Legitimate Predictions",
    metrics["legitimate_predictions"],
)

col4.metric(
    "Failed Predictions",
    metrics["failed_predictions"],
)


col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Fraud Prediction Rate",
    f"{metrics['fraud_prediction_rate']:.2%}",
)

col6.metric(
    "Average Fraud Probability",
    f"{metrics['average_fraud_probability']:.2%}",
)

col7.metric(
    "Average Inference Latency",
    f"{metrics['average_inference_latency_ms']:.2f} ms",
)

total_attempts = (
    metrics["total_predictions"]
    + metrics["failed_predictions"]
)

failure_rate = (
    metrics["failed_predictions"] / total_attempts
    if total_attempts else 0
)

col8.metric(
    "Inference Failure Rate",
    f"{failure_rate:.2%}",
)


# -------------------------------------
# PREDICTION DISTRIBUTION
# -------------------------------------

st.divider()

st.subheader("Prediction Distribution")

distribution = pd.DataFrame({
    "Classification": [
        "Fraud",
        "Legitimate",
    ],
    "Count": [
        metrics["fraud_predictions"],
        metrics["legitimate_predictions"],
    ],
})

st.bar_chart(
    distribution,
    x="Classification",
    y="Count",
)


# -------------------------------------
# MONITORING ALERTS
# -------------------------------------

st.subheader("Monitoring Alerts")

if metrics["total_predictions"] < 20:

    st.info(
        "Insufficient prediction volume "
        "for meaningful monitoring alerts."
    )

else:

    if metrics["fraud_prediction_rate"] > 0.20:

        st.warning(
            "Fraud prediction rate exceeds "
            "the demonstration threshold of 20%."
        )

    if metrics["average_inference_latency_ms"] > 500:

        st.warning(
            "Average model inference latency "
            "exceeds 500 milliseconds."
        )


if metrics["failed_predictions"] > 0:

    st.warning(
        "Model inference failures detected."
    )


# -------------------------------------
# RAW MONITORING METRICS
# -------------------------------------

with st.expander("View Raw Monitoring Metrics"):

    st.json(metrics)


st.divider()

st.caption(
    "Metrics are process-local and reset when "
    "the API restarts. Predicted fraud rates "
    "do not represent confirmed fraud rates."
)
