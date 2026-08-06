import os
import socket
import struct
from datetime import datetime
import subprocess
import inspect

from components import bq32002
from . import Logger, LogMsg
from .network import Network


class SystemTimeManager():
    def __init__(self) -> None:
        self.__logger = Logger()

    def get_timestamp(self):
        if Network().refresh_internet_connection_state()[0]:
            timestamp = self.get_time_from_ntp()
            bq32002.BQ32002().write_time(timestamp)
        else:
            timestamp = bq32002.BQ32002().read_time()
        return timestamp

    def set_system_time_with_timestamp(self, timestamp):
        try:
            dt = datetime.fromtimestamp(
                timestamp).strftime("%Y-%m-%d %H:%M:%S")
            cmd = f"date -s '{dt}'"
            try:
                subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                self.__logger.error(
                    LogMsg(
                        content=f"Set system time failed, attempting to escalate privileges using root account",
                        module=self.__module__,
                        filename=os.path.basename(os.path.abspath(__file__)),
                        currentframe=inspect.currentframe(),
                        pid=os.getpid()  # type: ignore
                    )
                )

            else:
                self.__logger.info(
                    LogMsg(
                        content=f"System time has been setted to : {dt}",
                        module=self.__module__,
                        filename=os.path.basename(os.path.abspath(__file__)),
                        currentframe=inspect.currentframe(),
                        pid=os.getpid()  # type: ignore
                    )
                )
            return True

        except subprocess.CalledProcessError as e:
            self.__logger.error(
                LogMsg(
                    content=f"An failure occurred while setting time: {e}",
                    module=self.__module__,
                    filename=os.path.basename(os.path.abspath(__file__)),
                    currentframe=inspect.currentframe(),
                    pid=os.getpid()  # type: ignore
                )
            )
            return False
        except Exception as e:
            self.__logger.error(
                LogMsg(
                    content=f"An failure occurred while setting time: {e}",
                    module=self.__module__,
                    filename=os.path.basename(os.path.abspath(__file__)),
                    currentframe=inspect.currentframe(),
                    pid=os.getpid()  # type: ignore
                )
            )
            return False

    def get_time_from_ntp(self, ntp_server: str = 'ntp1.aliyun.com', timeout: int = 5) -> int | None:
        try:
            ntp_packet = bytearray(48)
            ntp_packet[0] = 0x1B  # LI, Version, Mode

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(timeout)
                sock.sendto(ntp_packet, (ntp_server, 123))
                data, _ = sock.recvfrom(1024)

                if len(data) >= 48:
                    ntp_timestamp = struct.unpack('!12I', data)[10]
                    ntp_epoch = 2208988800
                    unix_timestamp = ntp_timestamp - ntp_epoch
                    self.__logger.info(
                        LogMsg(
                            content=f"NTP request time {datetime.fromtimestamp(unix_timestamp)}",
                            module=self.__module__,
                            filename=os.path.basename(
                                os.path.abspath(__file__)),
                            currentframe=inspect.currentframe(),
                            pid=os.getpid()  # type: ignore
                        )
                    )

                    return unix_timestamp

        except Exception as e:
            self.__logger.error(
                LogMsg(
                    content=f"NTP request failed {e}",
                    module=self.__module__,
                    filename=os.path.basename(os.path.abspath(__file__)),
                    currentframe=inspect.currentframe(),
                    pid=os.getpid()  # type: ignore
                )
            )
            return None


# 使用示例
if __name__ == "__main__":
    '''ntp_time = get_ntp_time('ntp1.aliyun.com')
    if ntp_time:
        print(f"Aliyun NTP 时间: {ntp_time}")
        print(f"本地时间: {datetime.now(timezone.utc)}")
        print(
            f"时间差: {(datetime.now(timezone.utc) - ntp_time).total_seconds():.3f} 秒")'''
