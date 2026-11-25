
import os

import httpx
import pandas as pd
import streamlit as st

from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Prediction History",
    page_icon="📊",
    layout="wide",
)

API_URL = os.getenv(
    "FRAUDLENS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_KEY = os.getenv(
    "FRAUDLENS_API_KEY"
)

st.title("FraudLens AI — Prediction History")

st.caption(
    "Persistent prediction analytics powered by PostgreSQL"
)

if not API_KEY:
    st.error("API key is not configured.")
    st.stop()

HEADERS = {
    "X-API-Key": API_KEY
}


@st.cache_data(
    ttl=15,
    show_spinner=False,
)
def fetch_history(limit):

    with httpx.Client(
        base_url=API_URL,
        headers=HEADERS,
        timeout=120,
    ) as client:

        summary = client.get(
            "/predictions/summary"
        )

        summary.raise_for_status()

        history = client.get(
            "/predictions/history",
            params={"limit": limit},
        )

        history.raise_for_status()

        return (
            summary.json(),
            history.json()["predictions"],
        )


with st.sidebar:

    st.header("History Settings")

    limit = st.slider(
        "Number of records",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
    )

    selected_label = st.selectbox(
        "Prediction type",
        ["All", "Fraud", "Legitimate"],
    )

    if st.button("Refresh data"):

        fetch_history.clear()
        st.rerun()


try:

    with st.spinner(
        "Loading prediction history..."
    ):

        summary, records = fetch_history(
            limit
        )

except httpx.HTTPStatusError as exc:

    if exc.response.status_code == 503:

        st.warning(
            "Persistent database is unavailable."
        )

    elif exc.response.status_code == 401:

        st.error(
            "API authentication failed."
        )

    else:

        st.error(
            f"API returned status "
            f"{exc.response.status_code}"
        )

    st.stop()

except httpx.RequestError:

    st.error(
        "Unable to connect to FraudLens API."
    )

    st.stop()


# Historical summary

st.subheader("All-time prediction statistics")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Predictions",
    summary["total_predictions"],
)

c2.metric(
    "Fraud Predictions",
    summary["fraud_predictions"],
)

c3.metric(
    "Legitimate Predictions",
    summary["legitimate_predictions"],
)

c4.metric(
    "Predicted Fraud Rate",
    f"{summary['fraud_prediction_rate']:.2%}",
)

c5, c6 = st.columns(2)

c5.metric(
    "Average Fraud Score",
    f"{summary['average_fraud_probability']:.4f}",
)

c6.metric(
    "Average Inference Latency",
    f"{summary['average_latency_ms']:.2f} ms",
)


if not records:

    st.info(
        "No predictions have been stored yet."
    )

    st.stop()


df = pd.DataFrame(records)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    utc=True,
)

df = df.sort_values("created_at")

if selected_label != "All":

    df = df[
        df["label"] == selected_label
    ]


if df.empty:

    st.info(
        "No records match the selected filter."
    )

    st.stop()


st.subheader("Historical analytics")

left, right = st.columns(2)

with left:

    st.markdown("**Prediction distribution**")

    counts = (
        df["label"]
        .value_counts()
        .rename("Predictions")
    )

    st.bar_chart(counts)


with right:

    st.markdown("**Fraud probability over time**")

    st.line_chart(
        df.set_index("created_at")[
            ["fraud_probability", "threshold"]
        ]
    )


st.markdown("**Model inference latency**")

st.line_chart(
    df.set_index("created_at")[
        "latency_ms"
    ]
)


st.subheader("Prediction records")

display_columns = [
    "created_at",
    "request_id",
    "fraud_probability",
    "threshold",
    "label",
    "model_version",
    "latency_ms",
]

st.dataframe(
    df[display_columns].sort_values(
        "created_at",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label="Download Prediction History",
    data=df.to_csv(index=False),
    file_name="fraudlens_prediction_history.csv",
    mime="text/csv",
)

st.caption(
    "Predicted fraud labels are model outputs, "
    "not confirmed fraud outcomes. "
    "History may include synthetic test records."
)
