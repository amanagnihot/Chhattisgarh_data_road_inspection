# ============================================================
# utils/logger.py — Structured JSON logger
# ============================================================

import logging
import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for easy parsing/aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                log_obj[key] = value
        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Call once at module level:
        logger = get_logger(__name__)
    """
    settings = get_settings()
    logger   = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # ── Console handler ──────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # ── File handler (rotating) ───────────────────────────────
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"road_assessment_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger