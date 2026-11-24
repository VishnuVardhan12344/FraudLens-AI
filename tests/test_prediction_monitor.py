
import pytest

from fraudlens.prediction_monitor import PredictionMonitor


def test_initial_metrics():

    monitor = PredictionMonitor()
    metrics = monitor.get_metrics()

    assert metrics["total_predictions"] == 0
    assert metrics["failed_predictions"] == 0
    assert metrics["fraud_prediction_rate"] == 0


def test_prediction_metrics():

    monitor = PredictionMonitor()

    monitor.record_prediction(0.02, 0, 20, "test-001")
    monitor.record_prediction(0.85, 1, 40, "test-002")
    monitor.record_prediction(0.10, 0, 30, "test-003")

    metrics = monitor.get_metrics()

    assert metrics["total_predictions"] == 3
    assert metrics["fraud_predictions"] == 1
    assert metrics["legitimate_predictions"] == 2
    assert metrics["fraud_prediction_rate"] == pytest.approx(
        1 / 3, abs=0.000001
    )
    assert metrics["average_fraud_probability"] == pytest.approx(
        0.97 / 3, abs=0.000001
    )
    assert metrics["average_inference_latency_ms"] == 30


def test_failed_prediction():

    monitor = PredictionMonitor()

    monitor.record_failure("test-004", "ValueError")

    metrics = monitor.get_metrics()

    assert metrics["failed_predictions"] == 1
    assert metrics["total_predictions"] == 0


@pytest.mark.parametrize(
    "probability,prediction,latency",
    [
        (-0.1, 0, 10),
        (1.5, 1, 10),
        (0.5, 2, 10),
        (0.5, 1, -10),
        (float("nan"), 1, 10),
        (0.5, 1, float("inf"))
    ]
)
def test_invalid_prediction(probability, prediction, latency):

    monitor = PredictionMonitor()

    with pytest.raises(ValueError):
        monitor.record_prediction(
            probability,
            prediction,
            latency,
            "invalid-test"
        )

    assert monitor.get_metrics()["total_predictions"] == 0
