"""Structured key=value logging for the service and CLIs."""

from __future__ import annotations

import logging
import re

from opentelemetry import trace

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

# Values made only of these characters are written bare; anything else is quoted and escaped,
# so user text in a field can never break a line or forge another key=value pair.
_BARE_VALUE = re.compile(r"[\w.:/@+,-]+")


def _format_value(value: object) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if _BARE_VALUE.fullmatch(text) else repr(text)


class KeyValueFormatter(logging.Formatter):
    """Appends ``extra={...}`` fields to the message as ``key=value`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        if not extras:
            return base
        pairs = " ".join(f"{k}={_format_value(v)}" for k, v in extras.items())
        return f"{base} {pairs}"


class TraceContextFilter(logging.Filter):
    """Adds ``trace_id`` inside a sampled trace, so a log line leads to its trace in Langfuse."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = trace.get_current_span().get_span_context()
        if context.is_valid and context.trace_flags.sampled and not hasattr(record, "trace_id"):
            record.trace_id = format(context.trace_id, "032x")
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(TraceContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
