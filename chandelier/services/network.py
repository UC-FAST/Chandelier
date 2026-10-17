"""Internet-connectivity probe."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

DEFAULT_TEST_URL = "http://www.msftconnecttest.com/connecttest.txt"
DEFAULT_EXPECTED_TEXT = "Microsoft Connect Test"


class Network:
    def __init__(
        self,
        test_url: str = DEFAULT_TEST_URL,
        timeout: float = 0.5,
        expected_text: str = DEFAULT_EXPECTED_TEXT,
    ):
        self._test_url = test_url
        self._timeout = timeout
        self._expected_text = expected_text
        self._is_internet_connected = False

    @property
    def is_internet_connected(self) -> bool:
        return self._is_internet_connected

    def refresh_internet_connection_state(
        self,
    ) -> tuple[bool, float | None]:
        """Return ``(connected, rtt_seconds)`` and update the cached state."""

        try:
            response = requests.get(self._test_url, timeout=self._timeout)
        except requests.RequestException:
            logger.warning("network state disconnected")
            self._is_internet_connected = False
            return False, None

        connected = (
            response.status_code == 200
            and response.text == self._expected_text
        )
        self._is_internet_connected = connected
        if connected:
            logger.info("network state connected")
            return True, response.elapsed.total_seconds()

        logger.warning("network state disconnected")
        return False, None
