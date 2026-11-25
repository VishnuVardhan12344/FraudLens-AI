
import argparse
import json

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraudlens.drift import detect_feature_drift

from fraudlens.retraining import train_candidate


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--current",
        required=True,
    )

    parser.add_argument(
        "--labeled-data",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
    )

    args = parser.parse_args()

    reference = pd.read_csv(
        args.reference
    )

    current = pd.read_csv(
        args.current
    )

    report = detect_feature_drift(
        reference,
        current,
    )

    significant = report[
        report["status"] == "Significant"
    ]

    moderate = report[
        report["status"] == "Moderate"
    ]

    # Trigger when two or more features
    # show significant drift.

    retraining_required = len(significant) >= 2

    summary = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "reference_records": len(reference),
        "current_records": len(current),
        "total_features": len(report),
        "significant_features": len(significant),
        "moderate_features": len(moderate),
        "stable_features": int(
            (report["status"] == "Stable").sum()
        ),
        "maximum_psi": float(
            report["psi"].max()
        ),
        "retraining_required": retraining_required,
        "candidate": None,
        "action": "monitor",
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_DIR / "drift_report.csv",
        index=False,
    )

    print("\nMODEL DRIFT REPORT")
    print(report.to_string(index=False))

    print(
        "\nSignificant features:",
        len(significant),
    )

    if retraining_required:

        print("\nALERT: Significant drift detected")

        summary["action"] = "retraining_recommended"

        if args.retrain:

            if not args.labeled_data:

                summary["action"] = (
                    "waiting_for_labeled_data"
                )

                print(
                    "Retraining blocked: "
                    "labeled data is required."
                )

            else:

                try:

                    summary["candidate"] = (
                        train_candidate(
                            args.labeled_data
                        )
                    )

                    summary["action"] = (
                        "candidate_trained"
                    )

                except Exception as exc:

                    summary["action"] = (
                        "retraining_failed"
                    )

                    summary["error_type"] = (
                        type(exc).__name__
                    )

                    raise

                finally:

                    save_summary(summary)

    else:

        print(
            "\nPASS: No retraining trigger"
        )

    save_summary(summary)

    print("\nPIPELINE COMPLETED")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


def save_summary(summary):

    path = REPORT_DIR / "drift_summary.json"

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
