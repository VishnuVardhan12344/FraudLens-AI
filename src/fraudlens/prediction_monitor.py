
import math
from threading import Lock

from fraudlens.logger import get_logger


logger = get_logger("fraudlens.prediction")


class PredictionMonitor:

    def __init__(self):
        self._lock = Lock()

        self.total_predictions = 0
        self.fraud_predictions = 0
        self.legitimate_predictions = 0
        self.failed_predictions = 0
        self.total_probability = 0.0
        self.total_latency_ms = 0.0

    def record_prediction(
        self,
        probability,
        prediction,
        latency_ms,
        request_id
    ):
        if prediction not in (0, 1):
            raise ValueError("Invalid prediction")

        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")

        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("Invalid latency")

        with self._lock:
            self.total_predictions += 1

            if prediction == 1:
                self.fraud_predictions += 1
            else:
                self.legitimate_predictions += 1

            self.total_probability += probability
            self.total_latency_ms += latency_ms

        logger.info(
            "prediction_completed",
            extra={
                "context": {
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 2)
                }
            }
        )

    def record_failure(self, request_id, error_type):

        with self._lock:
            self.failed_predictions += 1

        logger.error(
            "prediction_failed",
            extra={
                "context": {
                    "request_id": request_id,
                    "error_type": error_type
                }
            }
        )

    def get_metrics(self):

        with self._lock:
            total = self.total_predictions

            return {
                "total_predictions": total,
                "fraud_predictions": self.fraud_predictions,
                "legitimate_predictions": self.legitimate_predictions,
                "failed_predictions": self.failed_predictions,
                "fraud_prediction_rate": round(
                    self.fraud_predictions / total if total else 0,
                    6
                ),
                "average_fraud_probability": round(
                    self.total_probability / total if total else 0,
                    6
                ),
                "average_inference_latency_ms": round(
                    self.total_latency_ms / total if total else 0,
                    2
                )
            }


prediction_monitor = PredictionMonitor()
