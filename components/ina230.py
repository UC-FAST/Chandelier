import inspect
import os
import threading

import smbus2

from utils import ConfigLoader, LogMsg, Logger, singleton


@singleton
class INA230:
    # Register address dictionary (UPPERCASE)
    __reg = {
        'CONFIG': 0x00,
        'SHUNTVOLT': 0x01,
        'BUSVOLT': 0x02,
        'POWER': 0x03,
        'CURRENT': 0x04,
        'CALIB': 0x05
    }

    # Configuration parameters dictionary (UPPERCASE)
    __config = {
        'AVG_MODE': 0,      # 0=1 sample, 1=4, 2=16, 3=64, 4=128, 5=256, 6=512, 7=1024
        # Bus voltage conversion time (0=140μs,1=204μs,2=332μs,3=588μs,4=1.1ms,5=2.116ms,6=4.156ms,7=8.244ms)
        'VBUS_CT': 4,
        'VSH_CT': 4,         # Shunt voltage conversion time (same as above)
        # Operating mode (7=continuous shunt and bus measurement)
        'MODE': 7
    }

    def __init__(
            self,
            bus_number=ConfigLoader()['sensor']['INA230']['bus'],
            address=0x40,
            shunt_resistance=ConfigLoader()[
                'sensor']['INA230']['shunt resistance'],
            max_expected_current=ConfigLoader()[
                'sensor']['INA230']['maximum expected current']
    ):
        """
        Initialize INA230 device
        :param busNum: I2C bus number (Raspberry Pi typically 1)
        :param address: I2C device address (0x40-0x4F)
        :param shuntResistance: Shunt resistance value (Ohms)
        :param maxExpectedCurrent: Maximum expected current (Amperes)
        """

        self.__bus = smbus2.SMBus(bus_number)
        self.__addr = address
        self.__shunt_resistance = shunt_resistance

        # Calculate calibration values (camelCase naming)
        self.__currentLsb = max_expected_current / 32768.0
        self.__powerLsb = 25.0 * self.__currentLsb
        self.__calibration = int(
            0.00512 / (self.__currentLsb * shunt_resistance))
        self.__writeWord(self.__reg['CALIB'], self.__calibration)
        # Configure device
        self.__configure()

        self.__logger = Logger()
        self.__logger.info(
            LogMsg(
                content=f'INA230 init finished I2C_bus={bus_number} addr={hex(self.__addr)} shunt_resistance={self.__shunt_resistance} max_expected_current={max_expected_current}',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()  # type: ignore
            )
        )

    def __configure(self):
        """Write configuration to CONFIG register"""
        config = (self.__config['AVG_MODE'] << 9) | \
                 (self.__config['VBUS_CT'] << 6) | \
                 (self.__config['VSH_CT'] << 3) | \
            self.__config['MODE']

        self.__writeWord(self.__reg['CONFIG'], config)

    def __writeWord(self, register, data):
        """Write 16-bit data to register (big-endian format)"""
        msb = (data >> 8) & 0xFF
        lsb = data & 0xFF
        self.__bus.write_i2c_block_data(self.__addr, register, [msb, lsb])

    def __readWord(self, register):
        """Read 16-bit data from register (big-endian format)"""
        data = self.__bus.read_i2c_block_data(self.__addr, register, 2)
        return (data[0] << 8) | data[1]

    def read_voltage(self):
        """Read bus voltage (Volts)"""
        rawVoltage = self.__readWord(self.__reg['BUSVOLT'])
        data = rawVoltage * 0.00125  # LSB = 1.25mV
        self.__logger.debug(
            LogMsg(
                content=f'INA230 read voltage={data:.3f}V',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()  # type: ignore
            )
        )
        return data

    def read_current(self):
        """Read current (Amperes)"""
        rawCurrent = self.__readWord(self.__reg['CURRENT'])
        # Handle signed value (two's complement)
        if rawCurrent > 0x7FFF:
            rawCurrent -= 0x10000
        data = rawCurrent * self.__currentLsb
        self.__logger.debug(
            LogMsg(
                content=f'INA230 read current={data:.3f}A',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()  # type: ignore
            )
        )
        return data

    def read_power(self):
        """Read power (Watts)"""
        rawPower = self.__readWord(self.__reg['POWER'])
        data = rawPower * self.__powerLsb
        self.__logger.debug(
            LogMsg(
                content=f'INA230 read power={data:.3f}W',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()  # type: ignore
            )
        )
        return data

    def read_shunt_voltage(self):
        """Read shunt voltage (Volts)"""
        rawShunt = self.__readWord(self.__reg['SHUNTVOLT'])
        # Handle signed value
        if rawShunt > 0x7FFF:
            rawShunt -= 0x10000
        data = rawShunt * 0.0000025  # LSB = 2.5μV

        self.__logger.debug(
            LogMsg(
                content=f'INA230 read shunt voltage={data*1000:.3f}mV',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()  # type: ignore
            )
        )
        return data

    def close(self):
        """Close I2C connection"""
        self.__bus.close()


# Example usage
if __name__ == "__main__":
    # Configuration parameters (camelCase naming)
    i2cBus = 1               # Raspberry Pi I2C bus number
    i2cAddress = 0x40        # INA230 address
    shuntResistance = 0.002     # Shunt resistance (Ohms)
    maxCurrent = 3.2          # Maximum expected current (Amperes)

    try:
        # Create INA230 instance
        ina230 = INA230(
            bus_number=1,
            shunt_resistance=0.002,
            max_expected_current=3
        )

        # Print configuration information
        print("INA230 Configuration:")
        print(f"  I2C Address: 0x{ina230.__addr:02X}")
        print(f"  Shunt Resistance: {ina230.__shunt_resistance} Ω")
        print(f"  Calibration Value: {ina230.__calibration}")
        print(f"  Current LSB: {ina230.__currentLsb:.8f} A/bit")
        print(f"  Power LSB: {ina230.__powerLsb:.8f} W/bit")

        # Read and display sensor data
        print("\nSensor Readings:")
        print(f"  Bus Voltage: {ina230.read_voltage():.3f} V")
        print(f"  Shunt Voltage: {ina230.read_shunt_voltage():.6f} V")
        print(f"  Current: {ina230.read_current():.3f} A")
        print(f"  Power: {ina230.read_power():.3f} W")

        # Close connection
        ina230.close()

    except Exception as e:
        print(f"Error: {str(e)}")
