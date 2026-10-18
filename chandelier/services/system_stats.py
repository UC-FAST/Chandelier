"""System, power and Wi-Fi statistics for the monitor screen."""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

_IPV4_PATTERN = r"\d{1,3}(?:\.\d{1,3}){3}"
_MAC_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}")
_MISSING_IPV4 = {"0.0.0.0", "127.0.0.1"}


def _format_duration(seconds: float) -> str:
    """Format an uptime in seconds as ``[Nd ]HH:MM:SS``."""

    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SystemStats:
    """Collect hardware, power and network statistics for the monitor.

    All hardware reads are best-effort: a missing sensor or command yields
    ``None`` for that value instead of taking the screen down.  Network data is
    cached for a couple of seconds because it may involve subprocess calls.
    """

    _NETWORK_CACHE_SECONDS = 5.0
    _INTERNET_CACHE_SECONDS = 15.0
    _CPU_SNAPSHOT_SECONDS = 1.0
    _THROTTLE_CACHE_SECONDS = 5.0

    def __init__(self, battery, power_monitor, rtc, network=None):
        self._battery = battery
        self._power_monitor = power_monitor
        self._rtc = rtc
        self._network = network
        self._network_cache: dict[str, object] | None = None
        self._network_cached_at = 0.0
        self._internet_connected: bool | None = None
        self._internet_checked_at = 0.0
        self._internet_probe: threading.Thread | None = None
        self._cpu_snapshot: tuple[float | None, float | None] | None = None
        self._cpu_snapshot_at = 0.0
        self._throttle: str | None = None
        self._throttle_at = 0.0
        self._net_counters = None
        self._net_counters_at = 0.0

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    def hardware_report(self) -> dict[str, object]:
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil is a requirement
            logger.warning("psutil is not installed")
            return {
                "CPU {}%": None,
                "MEM {}%": None,
                "DISK {}%": None,
                "TEMP {} C": None,
                "System Time {}": datetime.now(),
                "RTC Time {}": None,
            }

        try:
            rtc_time = datetime.fromtimestamp(self._rtc.read_time())
        except (ValueError, OSError, OverflowError):
            rtc_time = None

        try:
            # interval=None is non-blocking: it reports usage since the
            # previous call, which the monitor makes twice per second.
            cpu_percent = round(psutil.cpu_percent(interval=None), 2)
        except Exception:
            logger.debug("CPU usage unavailable", exc_info=True)
            cpu_percent = None

        try:
            memory = psutil.virtual_memory()
            memory_percent = round(memory.used / memory.total * 100, 2)
        except Exception:
            logger.debug("memory usage unavailable", exc_info=True)
            memory_percent = None

        try:
            disk_percent = psutil.disk_usage("/").percent
        except Exception:
            logger.debug("disk usage unavailable", exc_info=True)
            disk_percent = None

        try:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            cores = (
                " ".join(f"{round(value)}" for value in per_core) + "%"
                if per_core
                else None
            )
        except Exception:
            logger.debug("per-core CPU usage unavailable", exc_info=True)
            cores = None

        try:
            swap_percent = round(psutil.swap_memory().percent, 2)
        except Exception:
            logger.debug("swap usage unavailable", exc_info=True)
            swap_percent = None

        temperature = self._cpu_temperature(psutil)
        self._cpu_snapshot = (cpu_percent, temperature)
        self._cpu_snapshot_at = time.monotonic()

        report: dict[str, object] = {
            "CPU {}%": cpu_percent,
            "Cores {}": cores,
            "MEM {}%": memory_percent,
            "Swap {}%": swap_percent,
            "DISK {}%": disk_percent,
            "TEMP {} C": temperature,
            "Throttle {}": self._throttle_state(),
        }

        frequency = self._cpu_frequency(psutil)
        if frequency is not None:
            report["CPU Freq {}MHz"] = frequency

        uptime = self._uptime(psutil)
        if uptime is not None:
            report["Uptime {}"] = _format_duration(uptime)

        load = self._load_average(psutil)
        if load is not None:
            report["Load {}"] = " ".join(f"{value:.2f}" for value in load)

        report["System Time {}"] = datetime.now()
        report["RTC Time {}"] = rtc_time
        return report

    def cpu_snapshot(self) -> tuple[float | None, float | None]:
        """Return ``(cpu_percent, temperature_c)`` for the status bar.

        The camera calls this every frame, so samples are cached for one
        second.  ``hardware_report`` refreshes the same cache while the
        system monitor is open.
        """

        now = time.monotonic()
        if (
            self._cpu_snapshot is not None
            and now - self._cpu_snapshot_at < self._CPU_SNAPSHOT_SECONDS
        ):
            return self._cpu_snapshot

        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil is a requirement
            logger.warning("psutil is not installed")
            return (None, None)

        try:
            cpu_percent = round(psutil.cpu_percent(interval=None), 1)
        except Exception:
            logger.debug("CPU usage unavailable", exc_info=True)
            cpu_percent = None
        temperature = self._cpu_temperature(psutil)

        self._cpu_snapshot = (cpu_percent, temperature)
        self._cpu_snapshot_at = now
        return self._cpu_snapshot

    def _throttle_state(self) -> str:
        """Summarise ``vcgencmd get_throttled`` flags (cached)."""

        now = time.monotonic()
        if (
            self._throttle is not None
            and now - self._throttle_at < self._THROTTLE_CACHE_SECONDS
        ):
            return self._throttle

        output = self._run(["vcgencmd", "get_throttled"])
        match = re.search(r"throttled=(0x[0-9a-fA-F]+)", output or "")
        state = (
            self._format_throttle(int(match.group(1), 16))
            if match
            else "N/A"
        )
        self._throttle = state
        self._throttle_at = now
        return state

    @staticmethod
    def _format_throttle(flags: int) -> str:
        labels = []
        if flags & 0x1:
            labels.append("UV")
        if flags & 0x2:
            labels.append("FREQ")
        if flags & 0x4:
            labels.append("THROT")
        if flags & 0x8:
            labels.append("TEMP")
        if not labels:
            labels.append("OK")
        if flags & 0xF0000:
            labels.append("!")
        return " ".join(labels)

    @staticmethod
    def _cpu_temperature(psutil) -> float | None:
        sensors = getattr(psutil, "sensors_temperatures", None)
        if sensors is None:
            return None
        try:
            temperatures = sensors()
        except Exception:
            logger.debug("temperature sensors unavailable", exc_info=True)
            return None

        if not isinstance(temperatures, dict) or not temperatures:
            return None
        for name in ("cpu_thermal", "soc_thermal", "coretemp"):
            entries = temperatures.get(name)
            if entries:
                return entries[0].current
        for entries in temperatures.values():
            if entries:
                return entries[0].current
        return None

    @staticmethod
    def _cpu_frequency(psutil) -> int | None:
        reader = getattr(psutil, "cpu_freq", None)
        if reader is None:
            return None
        try:
            frequency = reader()
        except Exception:
            logger.debug("CPU frequency unavailable", exc_info=True)
            return None
        current = getattr(frequency, "current", None)
        if not current:
            return None
        return round(current)

    @staticmethod
    def _uptime(psutil) -> float | None:
        reader = getattr(psutil, "boot_time", None)
        if reader is None:
            return None
        try:
            return time.time() - reader()
        except Exception:
            logger.debug("uptime unavailable", exc_info=True)
            return None

    @staticmethod
    def _load_average(psutil) -> tuple[float, float, float] | None:
        reader = getattr(psutil, "getloadavg", None) or getattr(
            os, "getloadavg", None
        )
        if reader is None:
            return None
        try:
            values = reader()
        except (AttributeError, OSError):
            return None
        if len(values) < 3:
            return None
        return values[0], values[1], values[2]

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------
    def power_report(self) -> dict[str, object]:
        battery_voltage = self._safe(
            lambda: round(self._battery.get_bat(), 2), "battery voltage"
        )
        battery_percent = self._battery_percent(battery_voltage)

        return {
            "BAT {}v {}%": (battery_voltage, battery_percent),
            "Bus Voltage {}V": self._safe(
                lambda: round(self._power_monitor.read_voltage(), 2),
                "bus voltage",
            ),
            "Shunt Voltage {}mV": self._safe(
                lambda: round(
                    self._power_monitor.read_shunt_voltage() * 1000, 3
                ),
                "shunt voltage",
            ),
            "Current {}mA": self._safe(
                lambda: round(self._power_monitor.read_current() * 1000, 2),
                "current",
            ),
            "Power {}W": self._safe(
                lambda: round(self._power_monitor.read_power(), 2), "power"
            ),
        }

    def _battery_percent(self, voltage: float | None) -> float | None:
        # Prefer the fuel gauge's own state-of-charge register when the driver
        # provides one; fall back to the voltage curve used by the old screen.
        soc = getattr(self._battery, "soc", None)
        if callable(soc):
            percent = self._safe(
                lambda: round(float(soc()), 1), "battery state of charge"
            )
            if percent is not None:
                return percent
        if voltage is None:
            return None
        return self._safe(
            lambda: self._battery.get_battery_percent(voltage),
            "battery percentage",
        )

    @staticmethod
    def _safe(call: Callable[[], Any], label: str) -> Any:
        try:
            return call()
        except Exception:
            logger.debug("%s unavailable", label, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    def network_report(self) -> dict[str, object]:
        now = time.monotonic()
        if (
            self._network_cache is not None
            and now - self._network_cached_at < self._NETWORK_CACHE_SECONDS
        ):
            report = dict(self._network_cache)
        else:
            try:
                report = self._collect_network_report()
            except Exception:
                logger.exception("network report failed")
                report = {
                    "Interface {}": None,
                    "ESSID {}": None,
                    "IP {}": None,
                    "MAC {}": None,
                    "Frequency {}": None,
                    "Link Quality {}": None,
                    "Signal Level {}": None,
                    "RX/TX {}": None,
                }
            self._network_cache = report
            self._network_cached_at = now

        # The internet probe is deliberately outside the static Wi-Fi cache:
        # it runs in the background so a slow request can never block the
        # render loop.
        result = dict(report)
        if self._network is not None:
            result["Internet {}"] = self._internet_state()
        return result

    def _collect_network_report(self) -> dict[str, object]:
        interface, essid, frequency, quality, level = self._wireless_info()
        if interface is None:
            interface = self._preferred_interface()
        return {
            "Interface {}": interface,
            "ESSID {}": essid,
            "IP {}": self._ip_address(interface),
            "MAC {}": self._mac_address(interface),
            "Frequency {}": frequency,
            "Link Quality {}": quality,
            "Signal Level {}": level,
            "RX/TX {}": self._network_throughput(),
        }

    def _network_throughput(self) -> str | None:
        """Return ``RX/TX`` KiB/s since the previous network sample."""

        try:
            import psutil

            counters = psutil.net_io_counters()
        except Exception:
            return None

        now = time.monotonic()
        previous = self._net_counters
        elapsed = now - self._net_counters_at
        self._net_counters = counters
        self._net_counters_at = now
        if previous is None or elapsed <= 0:
            return None

        receive = (
            max(0, counters.bytes_recv - previous.bytes_recv)
            / elapsed
            / 1024
        )
        transmit = (
            max(0, counters.bytes_sent - previous.bytes_sent)
            / elapsed
            / 1024
        )
        return f"{receive:.0f}K/{transmit:.0f}K"

    def _wireless_info(
        self,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        """Return ``(interface, essid, frequency, quality, signal)``.

        ``iw`` is preferred (current Raspberry Pi OS); ``iwconfig`` and
        ``/proc/net/wireless`` are kept as fallbacks for older images.
        """

        # /proc/net/wireless and iw are available on current Raspberry Pi OS;
        # iwconfig is only spawned when the modern tools have no details.
        interface = self._interface_from_proc_wireless()
        if interface is None:
            interface = self._interface_from_iw_dev()
        if interface is None:
            interface = self._interface_from_ifconfig()
        if interface is None:
            interface = self._preferred_interface()

        link = self._parse_iw_link(
            self._run(["iw", "dev", interface, "link"]) if interface else None
        )
        iwconfig = {}
        if not link:
            iwconfig = self._parse_iwconfig(self._run(["iwconfig"]))
            if iwconfig.get("interface"):
                interface = iwconfig["interface"]

        return (
            interface,
            link.get("essid") or iwconfig.get("essid"),
            link.get("frequency") or iwconfig.get("frequency"),
            link.get("quality") or iwconfig.get("quality"),
            link.get("level") or iwconfig.get("level"),
        )

    @staticmethod
    def _parse_iw_link(text: str | None) -> dict[str, str | None]:
        if not text:
            return {}

        result: dict[str, str | None] = {}
        match = re.search(r"^\s*SSID:\s*(.+?)\s*$", text, re.MULTILINE)
        if match:
            result["essid"] = match.group(1)

        match = re.search(r"^\s*freq:\s*([\d.]+)", text, re.MULTILINE)
        if match:
            frequency = float(match.group(1))
            if frequency >= 1000:
                result["frequency"] = f"{frequency / 1000:.2f} GHz"
            else:
                result["frequency"] = f"{frequency:.0f} MHz"

        match = re.search(
            r"^\s*signal:\s*(-?\d+)\s*dBm", text, re.MULTILINE
        )
        if match:
            dbm = int(match.group(1))
            result["level"] = f"{dbm} dBm"
            # iwconfig reports link quality on a 0..70 scale; derive the same
            # scale from dBm so both code paths render consistently.
            quality = max(0, min(70, 2 * (dbm + 100)))
            result["quality"] = f"{quality}/70"
        return result

    @staticmethod
    def _parse_iwconfig(text: str | None) -> dict[str, str | None]:
        if not text:
            return {}

        result: dict[str, str | None] = {}
        match = re.search(r"^(\w+)\s+IEEE\s", text, re.MULTILINE)
        if match:
            result["interface"] = match.group(1)

        match = re.search(r'ESSID:"([^"]*)"', text)
        if match:
            result["essid"] = match.group(1)

        match = re.search(r"Frequency:([\d.]+\s*[GM]Hz)", text)
        if match:
            result["frequency"] = match.group(1)

        match = re.search(r"Link Quality[=:]\s*([\d/]+)", text)
        if match:
            result["quality"] = match.group(1)

        match = re.search(r"Signal level[=:]\s*(-?\d+)\s*dBm", text)
        if match:
            result["level"] = f"{match.group(1)} dBm"
        return result

    def _interface_from_iw_dev(self) -> str | None:
        text = self._run(["iw", "dev"])
        if not text:
            return None
        match = re.search(r"^\s*Interface\s+(\S+)", text, re.MULTILINE)
        return match.group(1) if match else None

    @staticmethod
    def _interface_from_proc_wireless() -> str | None:
        try:
            with open("/proc/net/wireless", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return None
        for line in lines:
            match = re.match(r"\s*(\w+):", line)
            if match:
                return match.group(1)
        return None

    def _interface_from_ifconfig(self) -> str | None:
        text = self._run(["ifconfig"])
        if not text:
            return None
        interfaces = [
            name
            for name in re.findall(
                r"^(\w+)(?::|\s+Link encap)", text, re.MULTILINE
            )
            if name != "lo"
        ]
        for name in interfaces:
            if name.startswith(("wlan", "wlp", "wl")):
                return name
        for name in interfaces:
            if name.startswith(("eth", "en")):
                return name
        return interfaces[0] if interfaces else None

    @staticmethod
    def _preferred_interface() -> str | None:
        try:
            import psutil

            names = list(psutil.net_if_addrs())
        except Exception:
            return None

        try:
            stats = psutil.net_if_stats() or {}
        except Exception:
            stats = {}

        def score(name: str) -> tuple[int, int, str]:
            if name.startswith(("wlan", "wlp", "wl")):
                group = 0
            elif name.startswith(("eth", "en")):
                group = 1
            elif name == "lo":
                group = 3
            else:
                group = 2
            entry = stats.get(name)
            is_up = 0 if getattr(entry, "isup", False) else 1
            return group, is_up, name

        candidates = [name for name in names if name != "lo"]
        return min(candidates, key=score) if candidates else None

    def _ip_address(self, interface: str | None) -> str | None:
        addresses: dict[str, list] = {}
        try:
            import psutil

            addresses = psutil.net_if_addrs()
        except Exception:
            logger.debug("psutil network addresses unavailable", exc_info=True)

        names = [interface] if interface else list(addresses)
        for name in names:
            for address in addresses.get(name, []):
                if (
                    address.family == socket.AF_INET
                    and address.address not in _MISSING_IPV4
                ):
                    return address.address

        # ifconfig fallback for images without a usable psutil backend.
        text = self._run(["ifconfig"])
        if text:
            if interface:
                match = re.search(
                    rf"^{re.escape(interface)}:.*?inet (?:addr:)?"
                    rf"({_IPV4_PATTERN})",
                    text,
                    re.MULTILINE | re.DOTALL,
                )
                if match:
                    return match.group(1)
            for match in re.finditer(
                rf"inet (?:addr:)?({_IPV4_PATTERN})", text
            ):
                candidate = match.group(1)
                if candidate not in _MISSING_IPV4:
                    return candidate
        return None

    @staticmethod
    def _mac_address(interface: str | None) -> str | None:
        if not interface:
            return None
        try:
            import psutil

            addresses = psutil.net_if_addrs().get(interface, [])
        except Exception:
            return None
        for address in addresses:
            value = address.address or ""
            if (
                _MAC_PATTERN.fullmatch(value)
                and value.lower() != "00:00:00:00:00:00"
            ):
                return value.upper()
        return None

    def _internet_state(self) -> str:
        now = time.monotonic()
        due = (
            self._internet_connected is None
            or now - self._internet_checked_at >= self._INTERNET_CACHE_SECONDS
        )
        if due and (
            self._internet_probe is None
            or not self._internet_probe.is_alive()
        ):
            self._internet_checked_at = now
            self._internet_probe = threading.Thread(
                target=self._probe_internet,
                daemon=True,
                name="internet-probe",
            )
            self._internet_probe.start()

        if self._internet_connected is None:
            return "checking"
        return "connected" if self._internet_connected else "offline"

    def _probe_internet(self) -> None:
        try:
            connected, _ = self._network.refresh_internet_connection_state()
            self._internet_connected = bool(connected)
        except Exception:
            logger.debug("internet probe failed", exc_info=True)
            self._internet_connected = False

    @staticmethod
    def _run(command: list[str]) -> str | None:
        """Run a probe command and return stdout, or ``None`` when absent."""

        try:
            completed = subprocess.run(
                command, capture_output=True, check=False, timeout=2
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            logger.debug("command %s unavailable", command, exc_info=True)
            return None

        output = completed.stdout
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return output or ""
