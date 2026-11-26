
from fraudlens.logger import get_logger


logger = get_logger("fraudlens.test")


logger.info("application_started")


logger.info(
    "prediction_completed",
    extra={
        "context": {
            "prediction": 0,
            "fraud_probability": 0.0021,
            "latency_ms": 42
        }
    }
)


logger.warning(
    "high_fraud_probability",
    extra={
        "context": {
            "fraud_probability": 0.95
        }
    }
)


logger.error("prediction_failed")


try:
    result = 10 / 0

except ZeroDivisionError:
    logger.exception("calculation_failed")
