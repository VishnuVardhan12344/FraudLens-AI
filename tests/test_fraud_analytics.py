
def calculate_fraud_rate(fraud_count, total_count):
    if total_count <= 0:
        raise ValueError("Total count must be positive")

    return fraud_count / total_count * 100


def test_fraud_rate():
    result = calculate_fraud_rate(
        fraud_count=5,
        total_count=100,
    )

    assert result == 5.0


def test_zero_transactions():
    import pytest

    with pytest.raises(ValueError):
        calculate_fraud_rate(0, 0)
