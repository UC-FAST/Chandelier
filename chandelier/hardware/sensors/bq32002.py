"""BQ32002 real-time clock driver."""

from __future__ import annotations

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class BQ32002:
    """Minimal BQ32002 RTC driver over I2C (SMBus).

    Time registers are BCD encoded.  :meth:`configure_device` clears the
    stop and oscillator-disable flags so :meth:`read_time` can use fixed
    bit masks.
    """

    # Register map and flag bits; time values are BCD, with flag/reserved
    # bits in the upper bits of the seconds, hours and month registers.
    REGISTERS = {
        "SECONDS": 0x00,
        "MINUTES": 0x01,
        "HOURS": 0x02,
        "DAY": 0x03,
        "DATE": 0x04,
        "MONTH": 0x05,
        "YEAR": 0x06,
        "CALIBRATION": 0x07,
        "CONTROL": 0x0E,
        "STATUS": 0x0F,
    }
    STOP_BIT = 0x80  # SECONDS bit 7: oscillator stop flag
    HOUR_12_24_FLAG = 0x02  # cleared to force 24-hour readings
    OSC_DISABLE = 0x08  # CONTROL bit 3: oscillator disabled
    CAL_SIGN = 0x80  # CALIBRATION bit 7: negative offset

    def __init__(self, bus_number: int = 1, address: int = 0x68):
        import smbus2

        self._bus = smbus2.SMBus(bus_number)
        self._address = address
        self.configure_device()
        logger.info(
            "BQ32002 initialised on I2C bus %s at %#x",
            bus_number,
            address,
        )

    def configure_device(self) -> None:
        """Ensure the clock runs in 24-hour mode with the oscillator enabled."""

        # These flags live on the backup battery, so they persist across
        # power cycles and must be cleared on every boot.
        seconds = self._bus.read_byte_data(
            self._address, self.REGISTERS["SECONDS"]
        )
        if seconds & self.STOP_BIT:
            self._bus.write_byte_data(
                self._address,
                self.REGISTERS["SECONDS"],
                seconds & ~self.STOP_BIT,
            )

        status = self._bus.read_byte_data(
            self._address, self.REGISTERS["STATUS"]
        )
        if status & self.HOUR_12_24_FLAG:
            self._bus.write_byte_data(
                self._address,
                self.REGISTERS["STATUS"],
                status & ~self.HOUR_12_24_FLAG,
            )

        control = self._bus.read_byte_data(
            self._address, self.REGISTERS["CONTROL"]
        )
        if control & self.OSC_DISABLE:
            self._bus.write_byte_data(
                self._address,
                self.REGISTERS["CONTROL"],
                control & ~self.OSC_DISABLE,
            )

    def read_time(self) -> float:
        """Return the RTC time as a Unix timestamp."""

        # One block read covers the seven consecutive time registers, so
        # fields are less likely to straddle a rollover.
        data = self._bus.read_i2c_block_data(
            self._address, self.REGISTERS["SECONDS"], 7
        )
        # Mask off the flag/reserved bits that share the upper bits of the
        # seconds, hours and month registers.
        seconds = self._bcd_to_dec(data[0] & 0x7F)
        minutes = self._bcd_to_dec(data[1])
        hours = self._bcd_to_dec(data[2] & 0x3F)
        day = self._bcd_to_dec(data[3])  # noqa: F841 - kept for clarity
        date = self._bcd_to_dec(data[4])
        month = self._bcd_to_dec(data[5] & 0x1F)
        year = self._bcd_to_dec(data[6]) + 2000

        # A naive datetime is interpreted in the system timezone, matching
        # write_time().
        value = datetime(year, month, date, hours, minutes, seconds)
        logger.debug("BQ32002 read time %s", value)
        return value.timestamp()

    def write_time(self, timestamp: float) -> None:
        """Set the RTC from a Unix timestamp."""

        value = datetime.fromtimestamp(timestamp)
        # Seven consecutive BCD bytes starting at SECONDS; isoweekday()
        # returns 1..7 (Monday..Sunday) and the year field is two digits.
        data = [
            self._dec_to_bcd(value.second) & 0x7F,
            self._dec_to_bcd(value.minute),
            self._dec_to_bcd(value.hour) & 0x3F,
            self._dec_to_bcd(value.isoweekday()),
            self._dec_to_bcd(value.day),
            self._dec_to_bcd(value.month),
            self._dec_to_bcd(value.year % 100),
        ]
        self._bus.write_i2c_block_data(
            self._address, self.REGISTERS["SECONDS"], data
        )
        logger.info("BQ32002 wrote time %s", value)

    @property
    def calibration(self) -> tuple[int, bool]:
        """Return ``(offset, negative)`` from the calibration register."""
        value = self._bus.read_byte_data(
            self._address, self.REGISTERS["CALIBRATION"]
        )
        negative = bool(value & self.CAL_SIGN)
        # The low bits hold the magnitude; negative offsets are biased by
        # 32, so raw 16 with the sign bit set means -16.
        calibration = value & 0x1F
        if negative:
            return calibration - 32, True
        return calibration, False

    @calibration.setter
    def calibration(self, value: int) -> None:
        """Set the calibration offset, which must be in -16..+15."""
        if not -16 <= value <= 15:
            raise ValueError("calibration must be between -16 and +15")
        if value < 0:
            # Negative values set CAL_SIGN and flip bit 5 to form the
            # chip's signed field.
            raw = (abs(value) | self.CAL_SIGN) ^ 0x20
        else:
            raw = value
        self._bus.write_byte_data(
            self._address, self.REGISTERS["CALIBRATION"], raw
        )
        logger.info("BQ32002 calibration set to %s", value)

    @staticmethod
    def _bcd_to_dec(value: int) -> int:
        """Convert one packed BCD byte to an integer."""
        return (value // 16) * 10 + (value & 0x0F)

    @staticmethod
    def _dec_to_bcd(value: int) -> int:
        """Pack an integer 0..99 into BCD."""
        return (value // 10) << 4 | (value % 10)

    def close(self) -> None:
        """Release the SMBus handle."""
        self._bus.close()


if __name__ == "__main__":
    # Manual smoke test: stamp the current time, wait, then read it back.
    rtc = BQ32002()
    rtc.write_time(time.time())
    time.sleep(3)
    print(f"RTC reports time: {rtc.read_time()}")
    print(f"Calibration: {rtc.calibration}")
