"""Hardware adapters.

Every concrete adapter imports its Raspberry Pi dependency inside ``__init__``
or the method that needs it.  Importing this package never touches GPIO, SPI,
I2C, the camera or the audio device.
"""
