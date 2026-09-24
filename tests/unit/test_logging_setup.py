import logging

from food_concierge.logging_setup import KeyValueFormatter, configure_logging


def _record(**extra: object) -> logging.LogRecord:
    record = logging.makeLogRecord({"name": "t", "levelname": "INFO", "msg": "hello"})
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_extras_are_appended_as_key_value_pairs() -> None:
    line = KeyValueFormatter("%(message)s").format(_record(items=50, source="catalog"))

    assert line == "hello items=50 source=catalog"


def test_values_with_spaces_are_quoted() -> None:
    line = KeyValueFormatter("%(message)s").format(_record(detail="two words"))

    assert line == "hello detail='two words'"


def test_record_without_extras_is_unchanged() -> None:
    assert KeyValueFormatter("%(message)s").format(_record()) == "hello"


def test_configure_logging_replaces_handlers_and_sets_level() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("debug")
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, KeyValueFormatter)
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
