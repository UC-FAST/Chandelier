
from concurrent.futures import thread
import logging
import math
from operator import mul
import os
from queue import Empty, Queue
import threading
import time
import multiprocessing
from enum import StrEnum
from dataclasses import dataclass
import types

import psutil

try:
    from .configLoader import ConfigLoader
except:
    from configLoader import ConfigLoader

from .decorators import singleton

"""
Console Level Options
0 : Only print errors and critical messages to console.
1 : Print all info, errors, and critical messages to console.
2 : Print all debug, info, errors, and critical messages to console.
"""


@dataclass
class LogMsg:
    content: str
    module: str
    filename: str
    currentframe: types.FrameType | None
    pid: int | None

    @classmethod
    def from_str(cls, content: str):
        return cls(
            content=content,
            module='LogMsg from str',
            filename='',
            currentframe=None,
            pid=None
        )


class ColorEnum(StrEnum):
    kColorReset = "\033[0m"
    kColorGreen = "\033[0;32m"
    kColorBrightRed = "\033[1;31m"
    kColorBrightGreen = "\033[1;32m"
    kColorBrightYellow = "\033[1;33m"
    kColorBrightBlue = "\033[1;34m"
    kColorBrightMagenta = "\033[1;35m"
    kColorBrightCyan = "\033[1;36m"
    kColorBrightWhite = "\033[1;37m"


@singleton
class Logger:
    def __init__(self) -> None:
        self.__severity = min(
            math.floor(ConfigLoader('./config.json')['debug_level']), 50
        )
        self.__boot_time = psutil.boot_time()

        self.__severity_colors = {
            logging.DEBUG: ColorEnum.kColorBrightCyan,
            logging.INFO: ColorEnum.kColorBrightGreen,
            logging.WARNING: ColorEnum.kColorBrightYellow,
            logging.ERROR: ColorEnum.kColorBrightRed,
            logging.FATAL: ColorEnum.kColorBrightMagenta,
        }

        self.__log_queue = Queue()
        self.__running = True

        self.__logger_init()

        self.__log_thread = threading.Thread(
            target=self.__process_logs,
            daemon=True
        )
        self.__log_thread.start()

    def __logger_init(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(level=self.__severity)
        logger_handler = logging.StreamHandler()
        logger_handler.setLevel(level=self.__severity)
        formatter = logging.Formatter('%(message)s')
        logger_handler.setFormatter(formatter)
        self.__logger.addHandler(logger_handler)

    @staticmethod
    def __convert_int_to_severity(level: int) -> str:
        return ['DEBUG', ' INFO', ' WARN', 'ERROR', 'FATAL'][level//10-1]

    @staticmethod
    def time_point_to_string(time_point):
        secs = int(time_point)
        nsecs = int((time_point - secs) * 1_000_000_000)

        hours = secs // 3600
        minutes = (secs // 60) % 60
        seconds = secs % 60

        return f"{hours}:{minutes:02d}:{seconds:02d}.{nsecs:09d}"

    def __logger_builder(self, severity: int, msg: LogMsg):
        return '[{time}] [{pid}] {level_color}{level}{reset_color} '\
            '{module_color}{module}{reset_color} '\
            '{file_color}{file}:{lineno}{reset_color} '\
            '{content}'.format(
                time=self.time_point_to_string(time.time()-self.__boot_time),
                pid=msg.pid,
                level=self.__convert_int_to_severity(severity),
                module=msg.module,
                file=msg.filename,
                lineno=msg.currentframe.f_lineno if msg.currentframe else None,
                content=msg.content,
                level_color=self.__severity_colors[severity],
                module_color=ColorEnum.kColorBrightWhite,
                file_color=ColorEnum.kColorBrightBlue,
                reset_color=ColorEnum.kColorReset
            )

    def __log_message(self, severity: int, formatted_msg: str):
        
        match severity:
            case logging.FATAL:
                self.__logger.fatal(formatted_msg)
            case logging.ERROR:
                self.__logger.error(formatted_msg)
            case logging.WARNING:
                self.__logger.warning(formatted_msg)
            case logging.INFO:
                self.__logger.info(formatted_msg)
            case logging.DEBUG:
                self.__logger.debug(formatted_msg)

    def __process_logs(self):
        while self.__running:
            try:
                log_task = self.__log_queue.get()
                if log_task is None:  # Shutdown signal
                    break
                severity, formatted_msg = log_task
                self.__log_message(severity, formatted_msg)
            except Exception as e:
                try:
                    print(f"Logging error: {e}")
                except:
                    pass

    def __enqueue_log(self, severity: int, content: LogMsg):
        if not self.__running:
            return
        try:
            formatted_msg = self.__logger_builder(
                severity=severity, msg=content)
            self.__log_queue.put((severity, formatted_msg))
        except Exception as e:
            try:
                print(f"Failed to enqueue log: {e}")
            except:
                pass

    def debug(self, content: LogMsg):
        self.__enqueue_log(logging.DEBUG, content)

    def info(self, content: LogMsg):
        self.__enqueue_log(logging.INFO, content)

    def warning(self, content: LogMsg):
        self.__enqueue_log(logging.WARNING, content)

    def error(self, content: LogMsg):
        self.__enqueue_log(logging.ERROR, content)

    def fatal(self, content: LogMsg):
        self.__enqueue_log(logging.FATAL, content)

    def shutdown(self):
        self.__running = False
        self.__log_queue.put(None)
        if self.__log_thread.is_alive():
            self.__log_thread.join(timeout=5.0)

    def __del__(self):
        self.shutdown()


if __name__ == '__main__':
    l = Logger()
    l.debug(LogMsg.from_str('debug'))
    l.info(LogMsg.from_str('info'))
    l.warning(LogMsg.from_str('warning'))
    l.error(LogMsg.from_str('error'))
    l.fatal(LogMsg.from_str('fatal'))
