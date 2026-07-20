from app.constants.platform_enum import PlatformEnum
from app.utils.check_platform import get_os_name
from app.core.config import settings

class DoorManager:

    def __init__(self):
        self.model_barrie = None
        self.platform_barrie = get_os_name()

        self.pin_open = -1
        self.pin_close = -1

        self._init_pin()

    def _init_pin(self):
        if self.platform_barrie == PlatformEnum.UBUNTU:
            from app.utils.open_door.arduion_barrie import arduino_barrie
            self.pin_open = 2
            self.pin_close = 4
            self.model_barrie = arduino_barrie

        if self.platform_barrie == PlatformEnum.ORANGE_PI:
            from app.utils.open_door.orangepi_gpio import gpio_barrie_orangepi
            self.pin_open = 3
            self.pin_close = 5
            self.model_barrie = gpio_barrie_orangepi

        if self.platform_barrie == PlatformEnum.ORANGE_PI_MAX:
            from app.utils.open_door.orangepi_max_gpio import gpio_orange_pi_max_barrie
            self.pin_open = 3
            self.pin_close = 5
            self.model_barrie = gpio_orange_pi_max_barrie
        if self.platform_barrie == PlatformEnum.RADXA_E54C:
            from app.utils.open_door.gpio_barrie_radxa import gpi_radxa_barrie
            self.pin_open = 3
            self.pin_close = 5
            self.model_barrie = gpi_radxa_barrie

    def open_door(self, duration=0.5):
        if not settings.IS_OPEN_DOOR_WHEN_FACE_MASK:
            print("Door opening is disabled in settings.")
            return False
        if self.model_barrie is None:
            print("Barrie model not initialized")
            return False
        return self.model_barrie.open_barrie(self.pin_open, duration)

    def close_door(self, duration=0.5):
        if self.model_barrie is None:
            print("Barrie model not initialized")
            return False
        return self.model_barrie.open_barrie(self.pin_close, duration)

    def cleanup(self):
        if self.model_barrie is not None:
            self.model_barrie.cleanup()
        else:
            print("Barrie model not initialized for cleanup")


door_manager = DoorManager()
