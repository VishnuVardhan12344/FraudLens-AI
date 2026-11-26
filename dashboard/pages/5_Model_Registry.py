
import os

import httpx
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Model Registry",
    page_icon="📦",
    layout="wide",
)

API_URL = os.getenv(
    "FRAUDLENS_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_KEY = os.getenv("FRAUDLENS_API_KEY", "")

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5050",
)

MODEL_NAME = "FraudLens-RandomForest"

st.title("FraudLens AI — Model Registry")

st.caption(
    "Model versioning, deployment tracking "
    "and lifecycle governance"
)


@st.cache_data(ttl=15)
def fetch_model_info(api_url):

    response = httpx.get(
        f"{api_url}/model/info",
        headers={"X-API-Key": API_KEY},
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


if st.button("Refresh registry dashboard"):

    st.cache_data.clear()
    st.rerun()


st.subheader("Currently Deployed Model")

deployed_version = None

try:

    info = fetch_model_info(API_URL)

    deployed_version = str(
        info.get("registry_version", "unregistered")
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Model Version",
        deployed_version,
    )

    col2.metric(
        "Decision Threshold",
        f'{info["threshold"]:.2f}',
    )

    col3.metric(
        "Feature Count",
        info["feature_count"],
    )

    col4.metric(
        "API Status",
        "Online",
    )

    st.write("Model:", info.get("model"))

    st.write(
        "Deployed at:",
        info.get("deployed_at") or "Not recorded",
    )

except Exception as exc:

    st.warning(
        f"Model information unavailable: "
        f"{type(exc).__name__}"
    )


st.divider()

st.subheader("MLflow Model Registry")

connect_registry = st.checkbox(
    "Connect to local MLflow registry",
    value=False,
)

if connect_registry:

    try:

        from mlflow.tracking import MlflowClient

        client = MlflowClient(
            tracking_uri=TRACKING_URI
        )

        registered = client.get_registered_model(
            MODEL_NAME
        )

        aliases = registered.aliases

        champion = aliases.get("champion")
        candidate = aliases.get("candidate")
        previous = aliases.get("previous")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Champion",
            champion or "None",
        )

        col2.metric(
            "Candidate",
            candidate or "None",
        )

        col3.metric(
            "Previous",
            previous or "None",
        )

        if (
            champion is not None
            and deployed_version is not None
            and str(champion) != deployed_version
        ):

            st.warning(
                "Registry champion and deployed "
                "API version do not match."
            )

        versions = client.search_model_versions(
            f"name = '{MODEL_NAME}'"
        )

        rows = []

        for version in versions:

            assigned_aliases = [
                alias
                for alias, number in aliases.items()
                if str(number) == str(version.version)
            ]

            rows.append({
                "Version": str(version.version),
                "Aliases": ", ".join(assigned_aliases),
                "Status": version.status,
                "Role": version.tags.get(
                    "model_role", "Unknown"
                ),
                "Screening Passed": version.tags.get(
                    "screening_passed", "Unknown"
                ),
                "Approved": version.tags.get(
                    "approved", "Not recorded"
                ),
                "Run ID": version.run_id,
            })

        frame = pd.DataFrame(rows)

        st.subheader("Registered Model Versions")

        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
        )

        if not frame.empty:

            st.download_button(
                label="Download Registry Report",
                data=frame.to_csv(index=False),
                file_name="model_registry_report.csv",
                mime="text/csv",
            )

        st.success("MLflow registry connected")

    except ImportError:

        st.warning(
            "Install MLflow to access the local registry."
        )

    except Exception as exc:

        st.warning(
            f"Registry unavailable: "
            f"{type(exc).__name__}"
        )

else:

    st.info(
        "Enable the local registry connection "
        "to view registered model versions."
    )


st.divider()

st.subheader("Model Governance")

st.markdown("""
- **Champion:** Approved model selected for deployment.
- **Candidate:** Retrained model under evaluation.
- **Previous:** Former champion available for rollback.
- **Screening:** Minimum model performance requirements.
- **Approval:** Explicit authorization before promotion.
- **Deployment:** Installation of the selected model.
""")

st.caption(
    "Read-only governance dashboard. "
    "Promotion and rollback are performed "
    "through the management CLI."
)
