"""INA230 voltage/current/power monitor driver."""

from __future__ import annotations

import logging

from chandelier.config import Ina230Config

logger = logging.getLogger(__name__)


class INA230:
    """INA230 voltage/current/power monitor over I2C.

    The default address 0x40 assumes both address pins are grounded, which
    is the common breakout-board wiring.
    """

    REGISTERS = {
        "CONFIG": 0x00,
        "SHUNTVOLT": 0x01,
        "BUSVOLT": 0x02,
        "POWER": 0x03,
        "CURRENT": 0x04,
        "CALIB": 0x05,
    }
    # CONFIG fields: 1-sample average, 1.1 ms bus and shunt conversion
    # times, and continuous shunt+bus measurement mode.
    CONFIG = {
        "AVG_MODE": 0,
        "VBUS_CT": 4,
        "VSH_CT": 4,
        "MODE": 7,
    }

    def __init__(self, config: Ina230Config, address: int = 0x40):
        import smbus2

        self._bus = smbus2.SMBus(config.bus)
        self._address = address
        self._shunt_resistance = config.shunt_resistance
        # Current LSB = Imax / 2^15, so the signed register spans the
        # configured expected-current range.
        self._current_lsb = config.maximum_expected_current / 32768.0
        # The power register is scaled by a fixed 25x from the current LSB.
        self._power_lsb = 25.0 * self._current_lsb
        # Datasheet calibration constant: CAL = 0.00512 / (I_LSB * Rshunt).
        self._calibration = int(
            0.00512 / (self._current_lsb * config.shunt_resistance)
        )
        self._write_word(self.REGISTERS["CALIB"], self._calibration)
        self._configure()
        logger.info(
            "INA230 initialised on I2C bus %s at %#x (shunt=%s ohm)",
            config.bus,
            address,
            config.shunt_resistance,
        )

    def _configure(self) -> None:
        # CONFIG bitfields: AVG[11:9], VBUSCT[8:6], VSHCT[5:3], MODE[2:0].
        value = (
            (self.CONFIG["AVG_MODE"] << 9)
            | (self.CONFIG["VBUS_CT"] << 6)
            | (self.CONFIG["VSH_CT"] << 3)
            | self.CONFIG["MODE"]
        )
        self._write_word(self.REGISTERS["CONFIG"], value)

    def _write_word(self, register: int, value: int) -> None:
        """Write a big-endian 16-bit value to a register."""
        self._bus.write_i2c_block_data(
            self._address, register, [(value >> 8) & 0xFF, value & 0xFF]
        )

    def _read_word(self, register: int) -> int:
        """Read a big-endian 16-bit register value."""
        # A block read keeps the byte order explicit; the INA230 sends the
        # most significant byte first.
        data = self._bus.read_i2c_block_data(self._address, register, 2)
        return (data[0] << 8) | data[1]

    def read_voltage(self) -> float:
        """Bus voltage in volts."""

        # BUSVOLT LSB is 1.25 mV.
        value = self._read_word(self.REGISTERS["BUSVOLT"]) * 0.00125
        logger.debug("INA230 voltage %.3f V", value)
        return value

    def read_current(self) -> float:
        """Current in amperes (signed)."""

        raw = self._read_word(self.REGISTERS["CURRENT"])
        # CURRENT is a 16-bit two's-complement register.
        if raw > 0x7FFF:
            raw -= 0x10000
        value = raw * self._current_lsb
        logger.debug("INA230 current %.3f A", value)
        return value

    def read_power(self) -> float:
        """Power in watts."""

        # POWER is unsigned; its LSB is 25 * current LSB.
        value = self._read_word(self.REGISTERS["POWER"]) * self._power_lsb
        logger.debug("INA230 power %.3f W", value)
        return value

    def read_shunt_voltage(self) -> float:
        """Shunt voltage in volts (signed)."""

        raw = self._read_word(self.REGISTERS["SHUNTVOLT"])
        # SHUNTVOLT is 16-bit two's complement with a 2.5 uV LSB.
        if raw > 0x7FFF:
            raw -= 0x10000
        value = raw * 0.0000025
        logger.debug("INA230 shunt voltage %.3f mV", value * 1000)
        return value

    def close(self) -> None:
        """Release the SMBus handle."""
        self._bus.close()
