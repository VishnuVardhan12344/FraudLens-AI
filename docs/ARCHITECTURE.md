# FraudLens AI — Technical Architecture

## Objective

Build a modular fraud detection platform demonstrating data engineering, machine learning, API serving, monitoring and controlled model lifecycle management.

## Data Layer

Historical transaction records are ingested, validated and transformed into structured datasets. PySpark and Parquet demonstrate scalable processing and efficient analytical storage.

## Feature Engineering

The inference pipeline accepts 30 transaction features and generates four additional features:

- log_amount
- amount_is_zero
- elapsed_day
- elapsed_hour

The final model receives 34 features.

## Machine Learning Layer

Logistic Regression provides a baseline classification model. Random Forest provides a nonlinear classification approach.

Models are evaluated using precision, recall, F1 score and average precision. Decision thresholds are selected to balance fraud detection and false-positive costs.

## Inference Layer

FastAPI exposes authenticated REST endpoints for fraud predictions, health checks, model information and monitoring.

## Persistence Layer

PostgreSQL stores prediction metadata, including request identifiers, model version, predicted classification, probability and inference latency.

Raw transaction features are not stored in the prediction history table.

## Observability Layer

Prometheus metrics and structured JSON logs provide visibility into API requests, latency, prediction counts and service failures.

## Model Governance

MLflow manages model versions and champion, candidate and previous aliases.

Candidate models require performance screening and explicit approval before promotion. Deployment preserves a backup of the existing artifact.

## Deployment

Docker packages the inference service. GitHub Actions runs automated tests and Docker smoke tests.

Render hosts the cloud API, while Streamlit provides an interactive dashboard.

The registry and database integrations are currently demonstrated locally.

## Engineering Trade-offs

- Local SQLite simplifies MLflow setup but is not intended for distributed production registry operations.
- Process-local rate limiting is suitable for demonstration but requires shared state for distributed deployment.
- Batch drift detection is easier to validate than continuous streaming monitoring.
- Explicit model approval reduces the risk of uncontrolled model replacement.
- Local artifact backups support recovery but do not replace centralized versioned storage.