
#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

export PYTHONPATH="$ROOT/src"

PYTHON="$ROOT/.venv/bin/python"

"$PYTHON" scripts/run_drift_pipeline.py \
    --reference data/monitoring/reference.csv \
    --current data/monitoring/current_stable.csv \
    --labeled-data data/raw/creditcard.csv \
    --retrain
