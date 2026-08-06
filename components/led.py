import time
from utils import configLoader
import gpiozero

config = configLoader.ConfigLoader('config.json')



green = gpiozero.DigitalOutputDevice(pin=config['led']['led_green'], active_high=False, initial_value=False)
blue = gpiozero.DigitalOutputDevice(pin=config['led']['led_blue'], active_high=False, initial_value=False)

state = {
    green:False,
    blue:False
}

def toggle_state(led: gpiozero.output_devices.DigitalOutputDevice):
    state[led] = not state[led]
    led.toggle()


def on(led: gpiozero.output_devices.DigitalOutputDevice):
    state[led] = True
    led.on()


def off(led: gpiozero.output_devices.DigitalOutputDevice):
    state[led] = False
    led.off()


if __name__=='__main__':
    on(blue)
    #on(ledGreen)
    print(state)
    time.sleep(10)
