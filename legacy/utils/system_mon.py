import time
from components import ina230

if __name__=='__main__':
    a=ina230.INA230()
    while True:
        time.sleep(0.1)
        a.read_current()
        a.read_voltage()
        a.read_power()
