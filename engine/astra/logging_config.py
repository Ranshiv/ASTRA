"""Bounded local diagnostics and crash reports; no telemetry leaves the device."""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from . import config


def configure() -> None:
    config.PATHS.logs.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(config.PATHS.logs / "engine.log",
                                  maxBytes=5 * 1024 * 1024, backupCount=3,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(item, RotatingFileHandler) for item in root.handlers):
        root.addHandler(handler)

    def report(exc_type, exc, tb):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = config.PATHS.logs / f"crash-{stamp}.log"
        path.write_text("".join(traceback.format_exception(exc_type, exc, tb)),
                        encoding="utf-8")
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = report
