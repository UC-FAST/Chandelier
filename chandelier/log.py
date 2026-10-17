"""Logging setup.

The original project had a hand-written queue/thread logger with a ``LogMsg``
dataclass repeated at every call site.  The standard library logger is already
thread-safe, so this module only configures a console handler that keeps the
old visual format (elapsed time, pid, coloured level, module and source
location).
"""

from __future__ import annotations

import logging
import os
import time

_RESET = "\033[0m"
_COLORS = {
    logging.DEBUG: "\033[0;32m",       # green
    logging.INFO: "\033[1;32m",        # bright green
    logging.WARNING: "\033[1;33m",     # bright yellow
    logging.ERROR: "\033[1;31m",       # bright red
    logging.CRITICAL: "\033[1;35m",    # bright magenta
}
_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: " INFO",
    logging.WARNING: " WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "FATAL",
}

_PROCESS_START = time.monotonic()
_CONFIGURED = False


class ConsoleFormatter(logging.Formatter):
    """Console formatter that mirrors the previous project output."""

    def format(self, record: logging.LogRecord) -> str:
        severity = record.levelno
        if severity not in _COLORS:
            if severity >= logging.CRITICAL:
                severity = logging.CRITICAL
            elif severity >= logging.ERROR:
                severity = logging.ERROR
            elif severity >= logging.WARNING:
                severity = logging.WARNING
            elif severity >= logging.INFO:
                severity = logging.INFO
            else:
                severity = logging.DEBUG

        elapsed = time.monotonic() - _PROCESS_START
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        millis = int((elapsed - int(elapsed)) * 1000)
        stamp = f"{hours}:{minutes:02d}:{seconds:02d}.{millis:03d}"

        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        elif record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        return (
            f"[{stamp}] [{os.getpid()}] "
            f"{_COLORS[severity]}{_LEVEL_NAMES[severity]}{_RESET} "
            f"\033[1;37m{record.module}{_RESET} "
            f"\033[1;34m{record.filename}:{record.lineno}{_RESET} "
            f"{message}"
        )


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once.  ``level`` is clamped to 10..50."""

    global _CONFIGURED
    level = max(logging.DEBUG, min(int(level), logging.CRITICAL))

    root = logging.getLogger()
    root.setLevel(level)
    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ConsoleFormatter())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.  Safe to call before ``configure_logging``."""

    return logging.getLogger(name)


def verbose_console_to_level(verbose_console: int) -> int:
    """Map the old 0/1/2 picamera verbosity scale to a logging level."""

    if verbose_console >= logging.DEBUG:
        return int(verbose_console)
    return {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}.get(
        int(verbose_console), logging.INFO
    )
