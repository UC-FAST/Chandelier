import os
import requests
import inspect
import threading

from . import Logger, LogMsg


class Network():
    def __init__(self) -> None:
        self.__is_internet_connected: bool = False
        self.__logger = Logger()

    @property
    def is_internet_connected(self) -> bool:
        return self.__is_internet_connected

    def refresh_internet_connection_state(
            self,
            url: str = 'Http://www.msftconnecttest.com/connecttest.txt',
            timeout: float = 0.5,
            expected_text: str = "Microsoft Connect Test"
    ) -> tuple[bool, None | float]:
        """
            Refresh the object's notion of whether an internet connection is available by
            performing a short HTTP GET request to a connectivity-check endpoint.
            Parameters
            ----------
            self
                The instance whose __is_internet_connected attribute will be updated.
            url : str, optional
                The URL to request to determine connectivity. Default:
                'Http://www.msftconnecttest.com/connecttest.txt'.
            timeout : float, optional
                Maximum time in seconds to wait for the HTTP request. Default: 0.5.
            expected_text : str, optional
                Text expected to be returned by the endpoint to consider the connection
                valid. Default: "Microsoft Connect Test".
            Returns
            -------
            tuple[bool, float | None]
                A tuple (is_connected, rtt) where:
                - is_connected (bool): True if the request returned HTTP 200 and the
                  response body exactly equals expected_text; False otherwise.
                - rtt (float | None): Round-trip time of the request in seconds if
                  connected (computed as response.elapsed.microseconds / 1_000_000),
                  otherwise None.
            Side effects
            ------------
            Updates the instance attribute self.__is_internet_connected to True when a
            valid response is detected, or False on failure.
            Errors and exceptions
            ---------------------
            This method catches requests.ConnectionError, requests.Timeout and
            requests.RequestException and treats them as "not connected" (returning
            (False, None)). Other unexpected exceptions are not caught and will propagate.
            Notes
            -----
            - The comparison to expected_text is an exact string equality check (case
              and whitespace sensitive).
            - The RTT is derived from response.elapsed.microseconds which may differ
              slightly from requests' response.elapsed.total_seconds(); behavior is
              preserved from the original implementation.
            """
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200 and expected_text == response.text:
                self.__is_internet_connected = True
                self.__logger.info(
                    LogMsg(
                        content=f'Network state connected',
                        module=self.__module__,
                        filename=os.path.basename(os.path.abspath(__file__)),
                        currentframe=inspect.currentframe(),
                        pid=threading.current_thread().ident  # type: ignore
                    )
                )
                return True, response.elapsed.microseconds/1000000
            else:
                self.__logger.warning(
                    LogMsg(
                        content=f'Network state disconnected',
                        module=self.__module__,
                        filename=os.path.basename(os.path.abspath(__file__)),
                        currentframe=inspect.currentframe(),
                        pid=threading.current_thread().ident  # type: ignore
                    )
                )
                self.__is_internet_connected = False
                return False, None
        except (requests.ConnectionError, requests.Timeout, requests.RequestException):
            self.__logger.warning(
                LogMsg(
                    content=f'Network state disconnected',
                    module=self.__module__,
                    filename=os.path.basename(os.path.abspath(__file__)),
                    currentframe=inspect.currentframe(),
                    pid=threading.current_thread().ident  # type: ignore
                )
            )
            self.__is_internet_connected = False
            return False, None
