"""RTC/NTP/system-clock synchronisation."""

from __future__ import annotations

import logging
import socket
import struct
import subprocess
from datetime import datetime
from typing import Callable

from chandelier.services.network import Network

logger = logging.getLogger(__name__)

ClockSetter = Callable[[datetime], None]


class SystemTimeManager:
    def __init__(
        self,
        rtc,
        network: Network,
        ntp_server: str = "ntp1.aliyun.com",
        ntp_timeout: float = 5.0,
        clock_setter: ClockSetter | None = None,
    ):
        self._rtc = rtc
        self._network = network
        self._ntp_server = ntp_server
        self._ntp_timeout = ntp_timeout
        self._clock_setter = clock_setter or self._set_clock

    def get_timestamp(self) -> float:
        """Prefer NTP (and update the RTC); fall back to the RTC on failure."""

        if self._network.refresh_internet_connection_state()[0]:
            timestamp = self.get_time_from_ntp()
            if timestamp is not None:
                self._rtc.write_time(timestamp)
                return timestamp

        return self._rtc.read_time()

    def set_system_time_with_timestamp(self, timestamp: float) -> bool:
        value = datetime.fromtimestamp(timestamp)
        try:
            self._clock_setter(value)
        except Exception as exc:
            # The clock may already be kept in sync by systemd-timesyncd, and
            # changing it needs root; this is not fatal for the application.
            logger.warning("failed to set system time: %s", exc)
            return False
        logger.info("system time set to %s", value)
        return True

    @staticmethod
    def _set_clock(value: datetime) -> None:
        # Avoid shell=True: the formatted timestamp is passed as one argument.
        command = ["date", "-s", value.strftime("%Y-%m-%d %H:%M:%S")]
        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            return
        except FileNotFoundError:
            raise
        except subprocess.CalledProcessError:
            # Raspberry Pi OS gives the default ``pi`` user passwordless
            # sudo; try that before giving up.
            pass

        try:
            subprocess.run(
                ["sudo", "-n", *command],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("date command not found") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"date -s failed: {stderr or exc}"
            ) from exc

    def get_time_from_ntp(self) -> int | None:
        packet = bytearray(48)
        packet[0] = 0x1B  # LI, version, mode

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self._ntp_timeout)
                sock.sendto(packet, (self._ntp_server, 123))
                data, _ = sock.recvfrom(1024)
        except OSError:
            logger.exception("NTP request failed")
            return None

        if len(data) < 48:
            logger.warning("short NTP response (%d bytes)", len(data))
            return None

        ntp_timestamp = struct.unpack("!12I", data)[10]
        unix_timestamp = ntp_timestamp - 2_208_988_800
        logger.info(
            "NTP time %s", datetime.fromtimestamp(unix_timestamp)
        )
        return unix_timestamp
