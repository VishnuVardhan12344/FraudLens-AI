
from sqlalchemy import select, func

from fraudlens.database import SessionLocal
from fraudlens.db_models import PredictionRecord
from fraudlens.logger import get_logger


logger = get_logger("prediction_repository")


def save_prediction(
    request_id: str,
    probability: float,
    threshold: float,
    prediction: int,
    label: str,
    model_version: str,
    latency_ms: float,
):

    if SessionLocal is None:
        return False

    try:
        with SessionLocal() as session:

            record = PredictionRecord(
                request_id=request_id,
                fraud_probability=probability,
                threshold=threshold,
                prediction=prediction,
                label=label,
                model_version=model_version,
                latency_ms=latency_ms,
            )

            session.add(record)
            session.commit()

        logger.info(
            "prediction_saved",
            extra={
                "context": {
                    "request_id": request_id
                }
            },
        )

        return True

    except Exception as exc:

        logger.error(
            "prediction_storage_failed",
            extra={
                "context": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                }
            },
        )

        return False


def get_prediction_history(limit: int = 100):

    if SessionLocal is None:
        raise RuntimeError("Database not configured")

    limit = max(1, min(limit, 1000))

    with SessionLocal() as session:

        statement = (
            select(PredictionRecord)
            .order_by(
                PredictionRecord.created_at.desc(),
                PredictionRecord.id.desc(),
            )
            .limit(limit)
        )

        records = session.scalars(statement).all()

        return [
            {
                "id": record.id,
                "request_id": record.request_id,
                "fraud_probability": record.fraud_probability,
                "threshold": record.threshold,
                "prediction": record.prediction,
                "label": record.label,
                "model_version": record.model_version,
                "latency_ms": record.latency_ms,
                "created_at": record.created_at.isoformat(),
            }
            for record in records
        ]


def get_database_summary():

    if SessionLocal is None:
        raise RuntimeError("Database not configured")

    with SessionLocal() as session:

        total = session.scalar(
            select(func.count(PredictionRecord.id))
        ) or 0

        fraud = session.scalar(
            select(func.count(PredictionRecord.id))
            .where(PredictionRecord.prediction == 1)
        ) or 0

        legitimate = total - fraud

        average_probability = session.scalar(
            select(func.avg(
                PredictionRecord.fraud_probability
            ))
        ) or 0

        average_latency = session.scalar(
            select(func.avg(
                PredictionRecord.latency_ms
            ))
        ) or 0

        return {
            "total_predictions": total,
            "fraud_predictions": fraud,
            "legitimate_predictions": legitimate,
            "fraud_prediction_rate": (
                round(fraud / total, 6)
                if total else 0
            ),
            "average_fraud_probability": round(
                average_probability, 6
            ),
            "average_latency_ms": round(
                average_latency, 2
            ),
        }
