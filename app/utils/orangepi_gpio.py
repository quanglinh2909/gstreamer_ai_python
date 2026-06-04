import os
import threading
import time

try:
    import OPi.GPIO as GPIO
except ImportError:
    GPIO = None


class GPIOBarrieOrangePi:
    def __init__(self):
        # Set mode GPIO

        GPIO.set_custom_pin_mappings({
            3: 47,
            5: 46,
            7: 54,
            8: 131,
            10: 132,
            11: 138,
            12: 29,
            13: 139,
            15: 28,
            16: 59,
            18: 58,
            19: 49,
            21: 48,
            22: 92,
            23: 50,
            24: 52,
            26: 35
        })

        self.checkInput3 = True
        self.checkInput5 = True

        self.checkInput19 = True
        self.checkInput21 = True

        GPIO.setmode(GPIO.CUSTOM)
        # Set GPIO pin as output
        self.checkFist = False


    def open_barrie(self, io_pin):
        if self.checkFist == False:
            GPIO.setup(3, GPIO.OUT)
            GPIO.setup(5, GPIO.OUT)

            GPIO.output(3, GPIO.HIGH)
            GPIO.output(5, GPIO.HIGH)

            GPIO.setup(19, GPIO.OUT)
            GPIO.setup(21, GPIO.OUT)

            GPIO.output(19, GPIO.HIGH)
            GPIO.output(21, GPIO.HIGH)

            self.checkFist = True
        if io_pin == 3 and self.checkInput3 is True:
            self.checkInput3 = False
            threading.Thread(target=self.openTheading, args=(io_pin, self.setStatus3), daemon=True).start()

        if io_pin == 5 and self.checkInput5 is True:
            self.checkInput5 = False
            threading.Thread(target=self.openTheading, args=(io_pin, self.setStatus5), daemon=True).start()

        if io_pin == 19 and self.checkInput19 is True:
            self.checkInput19 = False
            threading.Thread(target=self.openTheading, args=(io_pin, self.setStatus19), daemon=True).start()

        if io_pin == 21 and self.checkInput21 is True:
            self.checkInput21 = False
            threading.Thread(target=self.openTheading, args=(io_pin, self.setStatus21), daemon=True).start()

    def setStatus3(self):
        self.checkInput3 = True

    def setStatus5(self):
        self.checkInput5 = True

    def setStatus19(self):
        self.checkInput19 = True

    def setStatus21(self):
        self.checkInput21 = True

    def openTheading(self, io_pin, setStatus):
        # Turn on the IO pin
        GPIO.output(io_pin, GPIO.LOW)
        time.sleep(0.5)  # Chờ 1 giây

        # Turn off the IO pin
        GPIO.output(io_pin, GPIO.HIGH)
        setStatus()

    def cleanup(self):
        pass


gpio_barrie_orangepi = GPIOBarrieOrangePi()
