"""Structured key=value logging for the service and CLIs."""

from __future__ import annotations

import logging

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class KeyValueFormatter(logging.Formatter):
    """Appends ``extra={...}`` fields to the message as ``key=value`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        if not extras:
            return base
        pairs = " ".join(f"{k}={v!r}" if isinstance(v, str) and " " in v else f"{k}={v}" for k, v in extras.items())
        return f"{base} {pairs}"


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
