"""Chandelier: Raspberry Pi camera terminal with sound-source localisation.

This package intentionally has no import-time side effects.  Hardware
libraries (gpiozero, picamera2, spidev, smbus2, pyaudio, ...) are imported
lazily by their adapters so that the package can be imported and unit-tested
on a development machine.
"""

__version__ = "0.2.0"
