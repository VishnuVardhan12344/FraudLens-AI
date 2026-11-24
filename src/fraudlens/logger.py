
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):

    def format(self, record):

        log_record = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": "fraudlens-api",
            "logger": record.name,
            "event": record.getMessage()
        }

        if hasattr(record, "context"):
            log_record["context"] = record.context

        if record.exc_info:
            log_record["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_record, default=str)


def get_logger(name="fraudlens"):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(JSONFormatter())

        logger.addHandler(handler)

    logger.propagate = False

    return logger
