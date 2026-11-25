
import json

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Model Drift",
    page_icon="📈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    ROOT / "reports" / "drift_report.csv"
)

SUMMARY_PATH = (
    ROOT / "reports" / "drift_summary.json"
)

st.title("FraudLens AI — Model Drift")

st.caption(
    "Feature stability and candidate retraining"
)

if st.sidebar.button("Refresh Report"):
    st.rerun()

if (
    not REPORT_PATH.exists()
    or not SUMMARY_PATH.exists()
):

    st.info(
        "Run the drift pipeline to generate reports."
    )

    st.stop()

report = pd.read_csv(REPORT_PATH)

summary = json.loads(
    SUMMARY_PATH.read_text()
)

st.caption(
    f"Report generated: {summary['timestamp']}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Features Monitored",
    summary["total_features"],
)

c2.metric(
    "Stable Features",
    summary["stable_features"],
)

c3.metric(
    "Significant Drift",
    summary["significant_features"],
)

c4.metric(
    "Maximum PSI",
    f"{summary['maximum_psi']:.4f}",
)

if summary["retraining_required"]:

    st.warning(
        "Significant feature drift detected. "
        "Candidate retraining is recommended."
    )

else:

    st.success(
        "No retraining trigger detected."
    )

st.subheader("Feature Stability Analysis")

chart = (
    alt.Chart(report)
    .mark_bar()
    .encode(
        x=alt.X(
            "psi:Q",
            title="Population Stability Index",
        ),
        y=alt.Y(
            "feature:N",
            sort="-x",
            title="Feature",
        ),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=[
                    "Stable",
                    "Moderate",
                    "Significant",
                ],
                range=[
                    "#16a34a",
                    "#eab308",
                    "#dc2626",
                ],
            ),
        ),
        tooltip=[
            "feature",
            "psi",
            "status",
        ],
    )
    .properties(height=650)
)

thresholds = pd.DataFrame({
    "threshold": [0.10, 0.25]
})

rules = (
    alt.Chart(thresholds)
    .mark_rule(
        strokeDash=[5, 5]
    )
    .encode(
        x="threshold:Q"
    )
)

st.altair_chart(
    chart + rules,
    use_container_width=True,
)

st.subheader("Feature Drift Report")

st.dataframe(
    report,
    hide_index=True,
    use_container_width=True,
)

st.download_button(
    "Download Drift Report",
    data=report.to_csv(index=False),
    file_name="fraudlens_drift_report.csv",
    mime="text/csv",
)

st.subheader("Automated Retraining")

st.write(
    "Pipeline action:",
    summary["action"],
)

candidate = summary.get("candidate")

if candidate:

    st.write(
        "Candidate version:",
        candidate["version"],
    )

    metrics = candidate["metrics"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precision",
        f"{metrics['precision']:.2%}",
    )

    c2.metric(
        "Recall",
        f"{metrics['recall']:.2%}",
    )

    c3.metric(
        "F1 Score",
        f"{metrics['f1']:.4f}",
    )

    c4.metric(
        "Average Precision",
        f"{metrics['average_precision']:.4f}",
    )

    st.metric(
        "Candidate Threshold",
        candidate["threshold"],
    )

    if candidate["passed_screening"]:

        st.success(
            "Candidate passed initial screening."
        )

    else:

        st.error(
            "Candidate failed initial screening."
        )

    st.info(
        "The deployed model remains unchanged. "
        "Candidate approval is required."
    )

else:

    st.info(
        "No candidate was trained in this run."
    )

st.caption(
    "Local batch monitoring report. "
    "Synthetic drift is not evidence of "
    "production model degradation."
)

