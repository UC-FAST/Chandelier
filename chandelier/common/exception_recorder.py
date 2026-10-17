"""Decorator that records uncaught exceptions and re-raises them."""

from __future__ import annotations

import logging
import time
import traceback
from functools import wraps
from typing import Any, Callable, Optional


def exception_recorder(
    path: str = "exceptions.log",
    logger: Optional[logging.Logger] = None,
    log_to_console: bool = True,
    include_args: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Log an exception with its traceback, then re-raise it.

    When ``logger`` is provided the record goes through it; otherwise it is
    appended to ``path``.  ``log_to_console`` also prints the record.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                lines = [
                    "=" * 50,
                    f"     time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f" function: {func.__name__}",
                    f"     file: {func.__code__.co_filename}:"
                    f"{func.__code__.co_firstlineno}",
                    f"exception: {type(exc).__name__}: {exc}",
                ]
                if include_args:
                    if args:
                        lines.append(f"positional args: {args}")
                    if kwargs:
                        lines.append(f"keyword args: {kwargs}")
                lines.append("traceback:")
                lines.extend(traceback.format_exc().splitlines())
                lines.append("=" * 50)
                message = "\n".join(lines) + "\n\n"

                if logger is not None:
                    logger.error(message)
                else:
                    try:
                        with open(path, "a", encoding="utf-8") as handle:
                            handle.write(message)
                    except OSError as write_error:
                        print(f"failed to write exception log: {write_error}")
                        print(message)

                if log_to_console:
                    print(message)
                raise

        return wrapper

    return decorator
