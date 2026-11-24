
import time
import uuid

from fastapi import Request
from starlette.responses import JSONResponse

from fraudlens.logger import get_logger

from fraudlens.metrics import (
    API_REQUESTS,
    API_LATENCY,
    RATE_LIMIT_REJECTIONS,
)


logger = get_logger("fraudlens.middleware")


async def request_monitoring_middleware(
    request: Request,
    call_next,
):

    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        "request_started",
        extra={
            "context": {
                "request_id": request_id,
                "method": request.method,
            }
        },
    )

    try:

        response = await call_next(request)

    except Exception as exc:

        logger.error(
            "request_failed",
            extra={
                "context": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                }
            },
        )

        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    duration_seconds = (
        time.perf_counter() - start_time
    )

    latency_ms = round(
        duration_seconds * 1000, 2
    )

    route = request.scope.get("route")

    route_path = getattr(
        route, "path", "unmatched"
    )

    response.headers["X-Request-ID"] = request_id

    response.headers["X-Response-Time-MS"] = str(
        latency_ms
    )

    # Prometheus API metrics
    if route_path != "/metrics":

        API_REQUESTS.labels(
            method=request.method
            if request.method in (
                "GET", "POST", "PUT", "PATCH",
                "DELETE", "HEAD", "OPTIONS"
            )
            else "OTHER",
            endpoint=route_path,
            status_code=str(response.status_code),
        ).inc()

        API_LATENCY.labels(
            endpoint=route_path
        ).observe(duration_seconds)

    if (
        route_path == "/predict"
        and response.status_code == 429
    ):
        RATE_LIMIT_REJECTIONS.inc()

    if response.status_code >= 500:

        log_method = logger.error

    elif response.status_code >= 400:

        log_method = logger.warning

    else:

        log_method = logger.info

    log_method(
        "request_completed",
        extra={
            "context": {
                "request_id": request_id,
                "method": request.method,
                "path": route_path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
        },
    )

    return response


