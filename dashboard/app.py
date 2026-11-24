
import os
from datetime import datetime
import httpx
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv


# -------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🛡️",
    layout="wide",
)
load_dotenv()

API_KEY = os.getenv("FRAUDLENS_API_KEY")
API_BASE_URL = os.getenv(
    "FRAUDLENS_API_URL",
    "https://fraudlens-ai-ecn5.onrender.com",
).rstrip("/")


# -------------------------------------
# 2. STORE RESULTS DURING THIS SESSION
# -------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# -------------------------------------
# SIDEBAR: DASHBOARD CONTROLS
# -------------------------------------

with st.sidebar:
    st.title("🛡️ FraudLens AI")

    st.caption("Fraud Detection Control Panel")

    st.divider()

    st.subheader("Model Information")
    st.write("Algorithm: Random Forest")
    st.write("Model: RandomForest-Optimized")
    st.write("Decision Threshold: 35%")

    st.divider()

    st.subheader("API Configuration")
    st.caption("Connected cloud endpoint:")
    st.code(API_BASE_URL)

    st.divider()

    st.subheader("Session Controls")

    if st.button(
        "Clear Prediction History",
        use_container_width=True,
    ):
        st.session_state.prediction_history = []
        st.rerun()

    st.caption(
        "History is stored only in this "
        "Streamlit session."
    )

# -------------------------------------
# 3. API COMMUNICATION FUNCTION
# -------------------------------------


def predict_transaction(transaction):

    if not API_KEY:
        raise ValueError(
            "FraudLens API key is not configured."
        )

    payload = {
        "features": transaction
    }

    headers = {
        "X-API-Key": API_KEY
    }

    with httpx.Client(timeout=120.0) as client:

        response = client.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers=headers,
        )

        response.raise_for_status()

        return response.json()



# -------------------------------------
# 4. DASHBOARD HEADER
# -------------------------------------

st.title("🛡️ FraudLens AI")

st.subheader("Fraud Detection Dashboard")

st.write(
    "Analyze synthetic transactions using "
    "a cloud-deployed Random Forest model."
)

st.info(
    "Model: RandomForest-Optimized | "
    "Decision threshold: 35%"
)


# -------------------------------------
# 5. DASHBOARD METRICS
# -------------------------------------

history = st.session_state.prediction_history

total_transactions = len(history)

fraud_alerts = sum(
    item["prediction"] == 1
    for item in history
)
average_fraud_probability = (
    sum(
        item["fraud_probability"]
        for item in history
    ) / total_transactions
    if total_transactions > 0
    else 0.0
)
alert_rate = (
    fraud_alerts / total_transactions * 100
    if total_transactions > 0
    else 0.0
)


# Four dashboard metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions Analyzed",
    total_transactions,
)

col2.metric(
    "Fraud Alerts",
    fraud_alerts,
)

col3.metric(
    "Fraud Alert Rate",
    f"{alert_rate:.2f}%",
)

col4.metric(
    "Average Fraud Probability",
    f"{average_fraud_probability * 100:.4f}%",
)


st.divider()


# -------------------------------------
# 6. TRANSACTION INPUT FORM
# -------------------------------------

st.header("Transaction Analysis")

st.caption(
    "Use synthetic test data only. "
    "Do not enter real payment-card details."
)

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:
        transaction_time = st.number_input(
            "Transaction Time (seconds)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

    with col2:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=120.0,
            step=10.0,
        )

    st.write("PCA Transaction Features")

    st.caption(
        "V1–V28 are anonymized numerical "
        "features from the fraud dataset."
    )

    feature_values = {}

    with st.expander(
        "Edit V1–V28 features",
        expanded=False,
    ):
        columns = st.columns(4)

        for i in range(1, 29):
            with columns[(i - 1) % 4]:
                feature_values[f"V{i}"] = (
                    st.number_input(
                        f"V{i}",
                        value=0.0,
                        format="%.6f",
                    )
                )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    )


# -------------------------------------
# 7. SEND TRANSACTION TO THE API
# -------------------------------------

if submitted:

    transaction = {
        "Time": float(transaction_time),
        "Amount": float(amount),
    }

    transaction.update(feature_values)

    try:
        with st.spinner(
            "Analyzing transaction..."
        ):
            result = predict_transaction(
                transaction
            )

        st.session_state.prediction_history.append(
        {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "amount": float(amount),
        "fraud_probability": float(
            result["fraud_probability"]
        ),
        "threshold": float(
            result["threshold"]
        ),
        "prediction": int(
            result["prediction"]
        ),
        "label": result["label"],
        "model": result["model"],
         }
        )

        st.subheader("Prediction Result")

        prediction = result["prediction"]

        if prediction == 1:
            st.error(
                "🚨 Potential Fraud Detected"
            )
        else:
            st.success(
                "✅ Transaction Classified "
                "as Legitimate"
            )

        probability = float(
            result["fraud_probability"]
        )

        threshold = float(
            result["threshold"]
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        result_col1.metric(
            "Fraud Probability",
            f"{probability * 100:.4f}%",
        )

        result_col2.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%",
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.json(result)

    except httpx.HTTPStatusError as error:
        st.error(
            f"API returned HTTP "
            f"{error.response.status_code}"
        )
        st.code(error.response.text)

    except httpx.RequestError as error:
        st.error(
            "Could not connect to the "
            "FraudLens AI cloud API."
        )
        st.write(str(error))

    except (KeyError, TypeError, ValueError) as error:
        st.error(
            "The API returned an unexpected "
            "prediction response."
        )
        st.write(str(error))

# -------------------------------------
# 8. TRANSACTION HISTORY
# -------------------------------------

st.divider()

st.header("Transaction History")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    # Convert probability to percentage.
    history_df["fraud_probability_pct"] = (
        history_df["fraud_probability"] * 100
    )

    # Select columns for the dashboard.
    display_df = history_df[
        [
            "timestamp",
            "amount",
            "fraud_probability_pct",
            "label",
            "model",
        ]
    ].copy()

    # Rename columns for readability.
    display_df.columns = [
        "Timestamp",
        "Amount",
        "Fraud Probability (%)",
        "Prediction",
        "Model",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(history_df)} predictions "
        "from the current session."
    )

else:

    st.info(
        "No transactions analyzed yet. "
        "Submit a transaction to see its history."
    )

# -------------------------------------
# 9. FRAUD PROBABILITY ANALYTICS
# -------------------------------------

st.divider()

st.header("Fraud Analytics")

history = st.session_state.prediction_history

if history:

    analytics_df = pd.DataFrame(history)

    # Assign a number to each transaction.
    analytics_df["transaction_id"] = range(
        1, len(analytics_df) + 1
    )

    # Convert probability to percentage.
    analytics_df["fraud_probability_pct"] = (
        analytics_df["fraud_probability"] * 100
    )

    analytics_df["threshold_pct"] = (
        analytics_df["threshold"] * 100
    )

    st.subheader(
        "Fraud Probability by Transaction"
    )

    probability_chart = (
        alt.Chart(analytics_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "transaction_id:O",
                title="Transaction Number",
            ),
            y=alt.Y(
                "fraud_probability_pct:Q",
                title="Fraud Probability (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.condition(
                alt.datum.prediction == 1,
                alt.value("#E74C3C"),
                alt.value("#2ECC71"),
            ),
            tooltip=[
                alt.Tooltip(
                    "transaction_id:O",
                    title="Transaction",
                ),
                alt.Tooltip(
                    "amount:Q",
                    title="Amount",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "fraud_probability_pct:Q",
                    title="Fraud Probability (%)",
                    format=".4f",
                ),
                alt.Tooltip(
                    "label:N",
                    title="Prediction",
                ),
            ],
        )
    )

    threshold_line = (
        alt.Chart(analytics_df)
        .mark_line(
            color="#F39C12",
            strokeDash=[6, 4],
        )
        .encode(
            x="transaction_id:O",
            y="threshold_pct:Q",
        )
    )

    st.altair_chart(
        probability_chart + threshold_line,
        use_container_width=True,
    )

else:
    st.info(
        "Analyze a transaction to display "
        "fraud probability charts."
    )

# -------------------------------------
# 10. FRAUD VS LEGITIMATE DISTRIBUTION
# -------------------------------------

st.subheader(
    "Fraud vs. Legitimate Transactions"
)

if history:

    distribution_df = (
        analytics_df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    distribution_chart = (
        alt.Chart(distribution_df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta(
                "count:Q",
                title="Transactions",
            ),
            color=alt.Color(
                "label:N",
                title="Prediction",
                scale=alt.Scale(
                    domain=[
                        "Legitimate",
                        "Fraud",
                    ],
                    range=[
                        "#2ECC71",
                        "#E74C3C",
                    ],
                ),
            ),
            tooltip=[
                "label:N",
                "count:Q",
            ],
        )
    )

    st.altair_chart(
        distribution_chart,
        use_container_width=True,
    )

else:
    st.info(
        "No prediction distribution "
        "available yet."
    )

# -------------------------------------
# 11. DOWNLOAD PREDICTION HISTORY
# -------------------------------------

st.divider()

st.header("Export Results")

if history:

    export_df = pd.DataFrame(history)

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction History CSV",
        data=csv_data,
        file_name="fraudlens_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Analyze transactions before "
        "downloading results."
    )
