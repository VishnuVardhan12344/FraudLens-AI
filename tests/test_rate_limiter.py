
import pytest

from fraudlens.rate_limiter import RateLimiter


def test_requests_allowed():

    limiter = RateLimiter(
        max_requests=3,
        window_seconds=60,
    )

    for _ in range(3):

        allowed, retry = limiter.allow_request()

        assert allowed is True
        assert retry == 0


def test_rate_limit_exceeded():

    limiter = RateLimiter(
        max_requests=2,
        window_seconds=60,
    )

    limiter.allow_request()
    limiter.allow_request()

    allowed, retry = limiter.allow_request()

    assert allowed is False
    assert 1 <= retry <= 60


def test_rate_limit_expiration():

    current_time = [0.0]

    limiter = RateLimiter(
        max_requests=2,
        window_seconds=60,
        clock=lambda: current_time[0],
    )

    assert limiter.allow_request()[0] is True
    assert limiter.allow_request()[0] is True
    assert limiter.allow_request()[0] is False

    current_time[0] = 61.0

    assert limiter.allow_request()[0] is True


def test_invalid_configuration():

    with pytest.raises(ValueError):

        RateLimiter(max_requests=0)
