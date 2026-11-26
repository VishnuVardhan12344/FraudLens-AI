# FraudLens AI — Operations Runbook

## Service Health

Check the local FastAPI health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Check MLflow:

```bash
curl http://127.0.0.1:5050/health
```

## Registry Status

```bash
PYTHONPATH=src python scripts/manage_models.py status
```

## Model Promotion

Review candidate metrics and confirm model readiness before executing:

```bash
PYTHONPATH=src python scripts/manage_models.py promote \
--version VERSION_NUMBER --approve
```

## Model Deployment

```bash
PYTHONPATH=src python scripts/manage_models.py deploy
```

Restart FastAPI after deployment and verify the authenticated model information endpoint.

## Model Rollback

Check the current champion and previous aliases before rollback.

```bash
PYTHONPATH=src python scripts/manage_models.py rollback
PYTHONPATH=src python scripts/manage_models.py deploy
```

Restart FastAPI and verify the restored version.

Do not repeat rollback blindly because the implementation swaps champion and previous aliases.

## Incident Troubleshooting

### MLflow connection refused

Verify that the MLflow server is running on port 5050 and is using the correct database.

### API authentication failure

Check the configured API key without exposing it in logs, screenshots or source control.

### Database unavailable

Check PostgreSQL container status, database configuration and network connectivity.

### Model artifact missing

Verify that the expected Joblib artifact exists and contains the required model bundle fields.

### Docker startup failure

Inspect container logs and confirm that dependencies, model artifacts and environment variables are configured correctly.

## Security Procedures

- Rotate exposed credentials immediately.
- Keep secrets outside version control.
- Load only trusted model artifacts.
- Review staged files before committing.
- Restrict local development services to localhost.
- Require successful tests before merging changes.