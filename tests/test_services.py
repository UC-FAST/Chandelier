import subprocess
import sys
import time
import types
from datetime import timedelta

import requests

from chandelier.services.network import Network
from chandelier.services.system_stats import SystemStats
from chandelier.services.time_sync import SystemTimeManager
from tests.fakes import FakeSensor


class FakeRtc:
    def __init__(self, timestamp=1000.0):
        self.timestamp = timestamp
        self.written = []

    def read_time(self):
        return self.timestamp

    def write_time(self, timestamp):
        self.written.append(timestamp)


class FakeNetwork:
    def __init__(self, connected=True):
        self.connected = connected

    def refresh_internet_connection_state(self):
        return self.connected, 0.01 if self.connected else None


def test_network_connected(monkeypatch):
    class Response:
        status_code = 200
        text = "Microsoft Connect Test"
        elapsed = timedelta(milliseconds=10)

    monkeypatch.setattr(
        "chandelier.services.network.requests.get", lambda *a, **k: Response()
    )
    network = Network()
    connected, rtt = network.refresh_internet_connection_state()
    assert connected is True
    assert rtt == 0.01
    assert network.is_internet_connected is True


def test_network_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(
        "chandelier.services.network.requests.get", fail
    )
    network = Network()
    assert network.refresh_internet_connection_state() == (False, None)
    assert network.is_internet_connected is False


def test_time_sync_prefers_ntp_and_updates_rtc(monkeypatch):
    rtc = FakeRtc()
    manager = SystemTimeManager(rtc, FakeNetwork(True))
    monkeypatch.setattr(manager, "get_time_from_ntp", lambda: 12345)
    assert manager.get_timestamp() == 12345
    assert rtc.written == [12345]


def test_time_sync_falls_back_to_rtc(monkeypatch):
    rtc = FakeRtc(777)
    manager = SystemTimeManager(rtc, FakeNetwork(True))
    monkeypatch.setattr(manager, "get_time_from_ntp", lambda: None)
    assert manager.get_timestamp() == 777


def test_time_sync_sets_clock_through_injected_setter():
    rtc = FakeRtc()
    values = []
    manager = SystemTimeManager(
        rtc, FakeNetwork(False), clock_setter=values.append
    )
    assert manager.set_system_time_with_timestamp(1000) is True
    assert len(values) == 1


def test_set_clock_falls_back_to_sudo(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "date":
            raise subprocess.CalledProcessError(
                1, command, stderr="permission denied"
            )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "chandelier.services.time_sync.subprocess.run", fake_run
    )
    manager = SystemTimeManager(
        FakeRtc(), FakeNetwork(False), clock_setter=SystemTimeManager._set_clock
    )

    assert manager.set_system_time_with_timestamp(1000) is True
    assert calls[0][0] == "date"
    assert calls[1][:3] == ["sudo", "-n", "date"]


def test_set_clock_failure_is_non_fatal(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="no")

    monkeypatch.setattr(
        "chandelier.services.time_sync.subprocess.run", fail
    )
    manager = SystemTimeManager(
        FakeRtc(), FakeNetwork(False), clock_setter=SystemTimeManager._set_clock
    )
    assert manager.set_system_time_with_timestamp(1000) is False


def test_power_report_with_fake_sensors():
    stats = SystemStats(FakeSensor(4.0), FakeSensor(), FakeSensor())
    report = stats.power_report()
    assert report["BAT {}v {}%"] == (4.0, 50.0)
    assert report["Bus Voltage {}V"] == 5.0
    assert report["Current {}mA"] == 500.0


def test_hardware_report_with_fake_psutil(monkeypatch):
    fake = types.ModuleType("psutil")
    fake.cpu_percent = lambda interval=None: 12.5
    fake.virtual_memory = lambda: types.SimpleNamespace(used=50, total=100)
    fake.disk_usage = lambda path: types.SimpleNamespace(percent=25.0)
    fake.sensors_temperatures = lambda: {
        "cpu_thermal": [types.SimpleNamespace(current=45.0)]
    }
    monkeypatch.setitem(sys.modules, "psutil", fake)

    stats = SystemStats(FakeSensor(), FakeSensor(), FakeSensor())
    report = stats.hardware_report()
    assert report["CPU {}%"] == 12.5
    assert report["MEM {}%"] == 50.0
    assert report["DISK {}%"] == 25.0
    assert report["TEMP {} C"] == 45.0


def test_network_report_internet_probe_does_not_block(monkeypatch):
    class SlowNetwork:
        def __init__(self):
            self.calls = 0

        def refresh_internet_connection_state(self):
            self.calls += 1
            time.sleep(0.1)
            return True, 0.01

    network = SlowNetwork()
    stats = SystemStats(
        FakeSensor(), FakeSensor(), FakeSensor(), network=network
    )
    monkeypatch.setattr(
        stats, "_collect_network_report", lambda: {"Interface {}": "wlan0"}
    )

    started = time.perf_counter()
    report = stats.network_report()
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    assert report["Internet {}"] == "checking"

    time.sleep(0.15)
    report = stats.network_report()
    assert report["Internet {}"] == "connected"
    assert network.calls == 1
