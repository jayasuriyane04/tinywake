"""
tinywake/logging_utils.py

Tiny structured-logging helper so every module prints in the same
"[TAG] message" style shown throughout the spec, e.g.:

    [KWS] score=0.93
    [WAKE] Hey Nova detected
    [STREAM] session=42 started
"""
from __future__ import annotations
import logging
import sys
import time

_LEVEL_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[36m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
}
_RESET = "\033[0m"


class TagFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", record.name.upper())
        color = _LEVEL_COLORS.get(record.levelname, "")
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        msg = record.getMessage()
        return f"{color}{ts} [{tag}] {msg}{_RESET}"


def get_logger(tag: str) -> logging.Logger:
    """Return a logger that always prints with the given [TAG] prefix."""
    logger = logging.getLogger(tag)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(TagFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    class _TagAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            kwargs.setdefault("extra", {})["tag"] = tag
            return msg, kwargs

    return _TagAdapter(logger, {})
