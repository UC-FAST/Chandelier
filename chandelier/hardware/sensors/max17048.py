"""MAX17048 battery fuel-gauge driver."""

from __future__ import annotations

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Open-circuit voltage to state-of-charge approximation for a single Li-ion
# cell.  Keys are in descending order so the lookup returns the first entry
# whose threshold is strictly below the measured voltage; no interpolation.
_VOLTAGE_TO_PERCENT = OrderedDict(
    {
        4.20: 100.0, 4.19: 99.5, 4.18: 99.0, 4.17: 98.0, 4.16: 97.0,
        4.15: 96.0, 4.14: 95.0, 4.13: 94.0, 4.12: 92.5, 4.11: 91.0,
        4.10: 89.5, 4.09: 88.0, 4.08: 86.5, 4.07: 85.0, 4.06: 83.5,
        4.05: 82.0, 4.04: 80.5, 4.03: 79.0, 4.02: 77.5, 4.01: 76.0,
        4.00: 74.5, 3.99: 73.0, 3.98: 71.5, 3.97: 70.0, 3.96: 68.5,
        3.95: 67.0, 3.94: 65.5, 3.93: 64.0, 3.92: 62.5, 3.91: 61.0,
        3.90: 59.5, 3.89: 58.0, 3.88: 56.5, 3.87: 55.0, 3.86: 53.5,
        3.85: 52.0, 3.84: 50.5, 3.83: 49.0, 3.82: 47.5, 3.81: 46.0,
        3.80: 44.5, 3.79: 43.0, 3.78: 41.5, 3.77: 40.0, 3.76: 38.5,
        3.75: 37.0, 3.74: 35.5, 3.73: 34.0, 3.72: 32.5, 3.71: 31.0,
        3.70: 29.0, 3.69: 27.0, 3.68: 25.0, 3.67: 23.0, 3.66: 21.0,
        3.65: 19.0, 3.64: 17.0, 3.63: 15.0, 3.62: 13.0, 3.61: 11.0,
        3.60: 9.0, 3.59: 7.5, 3.58: 6.0, 3.57: 5.0, 3.56: 4.0,
        3.55: 3.5, 3.54: 3.0, 3.53: 2.5, 3.52: 2.0, 3.51: 1.5,
        3.50: 1.0, 3.49: 0.8, 3.48: 0.6, 3.47: 0.4, 3.46: 0.2,
        3.45: 0.1, 3.44: 0.05, 3.43: 0.01, 3.27: 0.0,
    }
)


class MAX17048:
    """MAX17048 fuel gauge over I2C (default address 0x36).

    Register values are fixed point with datasheet-defined LSBs: 78.125 uV
    for voltage and 1/256 percent for state of charge.
    """

    # 16-bit big-endian registers; addresses from the datasheet.
    REGISTERS = {
        "VCELL": 0x02,
        "SOC": 0x04,
        "MODE": 0x06,
        "VERSION": 0x08,
        "HIBRT": 0x0A,
        "CONFIG": 0x0C,
        "VALRT": 0x14,
        "CRATE": 0x16,
        "VRESET": 0x18,
        "STATUS": 0x1A,
        "CMD": 0xFE,
    }

    def __init__(self, bus_number: int = 1, address: int = 0x36):
        import smbus2

        self._bus = smbus2.SMBus(bus_number)
        self._address = address
        logger.info(
            "MAX17048 initialised on I2C bus %s at %#x",
            bus_number,
            address,
        )

    def _read_word(self, register: int) -> int:
        """Read a big-endian 16-bit register value."""
        msb, lsb = self._bus.read_i2c_block_data(
            self._address, register, 2
        )
        return msb << 8 | lsb

    def get_bat(self) -> float:
        """Battery voltage in volts."""

        # VCELL LSB is 78.125 uV (1.25 mV / 16).
        value = self._read_word(self.REGISTERS["VCELL"]) * 0.000078125
        logger.debug("MAX17048 voltage %.3f V", value)
        return value

    def soc(self) -> float:
        """State of charge in percent."""

        # SOC LSB is 1/256 percent.
        return self._read_word(self.REGISTERS["SOC"]) * 3.90625e-05

    def version(self) -> int:
        """Return the chip version register (useful to probe the device)."""
        return self._read_word(self.REGISTERS["VERSION"])

    @staticmethod
    def get_battery_percent(voltage: float) -> float | None:
        """Approximate state of charge from a resting cell voltage.

        Returns ``None`` when the voltage is below the lowest tabulated
        value (deeply discharged or no cell connected).
        """
        for threshold, percent in _VOLTAGE_TO_PERCENT.items():
            if voltage > threshold:
                return percent
        return None

    def close(self) -> None:
        """Release the SMBus handle."""
        self._bus.close()
