#!/usr/bin/env python3
"""
GPIO Barrier Controller for Radxa E54C
Sử dụng thư viện gpiod (libgpiod) v2.x thay vì OPi.GPIO
"""

import os
import threading
import time

try:
    import gpiod
    from gpiod.line import Direction, Value

    GPIOD_VERSION = tuple(map(int, gpiod.__version__.split('.')))
except ImportError:
    print("❌ Lỗi: Chưa cài đặt thư viện gpiod")
    print("Cài đặt bằng: sudo apt install -y python3-libgpiod")
    exit(1)


class GPIOBarrieRadxa:
    """
    GPIO Barrier Controller cho Radxa E54C

    Pin mapping (Physical → Chip/Line → GPIO number):
    - PIN 3  → gpiochip4/line11 → GPIO 139
    - PIN 5  → gpiochip4/line10 → GPIO 138
    - PIN 19 → Cần xác định (chưa có trong gpioinfo)
    - PIN 21 → Cần xác định (chưa có trong gpioinfo)
    """

    # Định nghĩa mapping giữa physical pin và (chip, line)
    PIN_MAPPING = {
        3: {'chip': 4, 'line': 11},  # gpiochip4 line 11
        5: {'chip': 4, 'line': 10},  # gpiochip4 line 10
        11: {'chip': 0, 'line': 24},  # gpiochip0 line 24
        13: {'chip': 0, 'line': 23},  # gpiochip0 line 23
        # Thêm PIN 19, 21 nếu biết mapping
        # 19: {'chip': X, 'line': Y},
        # 21: {'chip': X, 'line': Y},
    }

    def __init__(self):
        """Khởi tạo GPIO controller"""
        self.chips = {}
        self.lines = {}

        # Trạng thái kiểm soát
        self.checkInput3 = True
        self.checkInput5 = True
        self.checkInput19 = True
        self.checkInput21 = True
        self.checkFirst = False

        # Mở GPIO chips
        self._init_gpio()

    def _init_gpio(self):
        """Khởi tạo GPIO chips và lines"""
        try:
            # Mở tất cả các chip cần thiết
            used_chips = set(pin_info['chip'] for pin_info in self.PIN_MAPPING.values())

            for chip_num in used_chips:
                chip_path = f"/dev/gpiochip{chip_num}"
                if os.path.exists(chip_path):
                    self.chips[chip_num] = gpiod.Chip(chip_path)
                    print(f"✓ Đã mở {chip_path}")
                else:
                    print(f"⚠️  Không tìm thấy {chip_path}")

        except Exception as e:
            print(f"❌ Lỗi khởi tạo GPIO: {e}")
            raise

    def _get_line(self, physical_pin):
        """
        Lấy GPIO line từ physical pin number (gpiod v2.x API)

        Args:
            physical_pin: Số physical pin (3, 5, 19, 21...)

        Returns:
            LineSettings and line offset
        """
        if physical_pin not in self.PIN_MAPPING:
            raise ValueError(f"Physical pin {physical_pin} chưa được định nghĩa trong PIN_MAPPING")

        # Kiểm tra cache
        if physical_pin in self.lines:
            return self.lines[physical_pin]

        # Lấy thông tin chip và line
        pin_info = self.PIN_MAPPING[physical_pin]
        chip_num = pin_info['chip']
        line_num = pin_info['line']

        if chip_num not in self.chips:
            raise RuntimeError(f"GPIO chip {chip_num} chưa được mở")

        # Request line với gpiod v2.x API
        chip = self.chips[chip_num]

        # Tạo line settings cho output
        line_settings = {
            line_num: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                output_value=Value.ACTIVE  # HIGH = 1
            )
        }

        # Request line
        line_request = chip.request_lines(
            consumer="gpio_barrier",
            config=line_settings
        )

        # Lưu vào cache
        self.lines[physical_pin] = {
            'request': line_request,
            'line_num': line_num
        }

        return self.lines[physical_pin]

    def open_barrie(self, io_pin, duration=0.5):
        """
        Mở barrier (kích hoạt GPIO pin)

        Args:
            io_pin: Số physical pin (3, 5, 19, 21)
            duration: Thời gian mở cửa (giây), mặc định 0.5s
        """
        # Khởi tạo tất cả GPIO lần đầu
        if not self.checkFirst:
            try:
                # Khởi tạo các pin có trong mapping
                for pin in [3, 5]:  # Chỉ dùng pin có sẵn trong mapping
                    if pin in self.PIN_MAPPING:
                        line_info = self._get_line(pin)
                        # Set HIGH = inactive
                        line_info['request'].set_value(line_info['line_num'], Value.ACTIVE)
                        print(f"✓ Đã khởi tạo physical pin {pin}")

                # Chú ý: PIN 19, 21 chưa có trong mapping, cần xác định trước
                # for pin in [19, 21]:
                #     if pin in self.PIN_MAPPING:
                #         line = self._get_line(pin)
                #         line.set_value(1)

                self.checkFirst = True
            except Exception as e:
                print(f"❌ Lỗi khởi tạo GPIO: {e}")
                return

        # Xử lý từng pin
        if io_pin == 3 and self.checkInput3:
            self.checkInput3 = False
            threading.Thread(
                target=self.open_threading,
                args=(io_pin, self.setStatus3, duration),
                daemon=True
            ).start()

        elif io_pin == 5 and self.checkInput5:
            self.checkInput5 = False
            threading.Thread(
                target=self.open_threading,
                args=(io_pin, self.setStatus5, duration),
                daemon=True
            ).start()

        elif io_pin == 19 and self.checkInput19:
            if 19 not in self.PIN_MAPPING:
                print(f"⚠️  Physical pin 19 chưa được định nghĩa trong PIN_MAPPING")
                return
            self.checkInput19 = False
            threading.Thread(
                target=self.open_threading,
                args=(io_pin, self.setStatus19, duration),
                daemon=True
            ).start()

        elif io_pin == 21 and self.checkInput21:
            if 21 not in self.PIN_MAPPING:
                print(f"⚠️  Physical pin 21 chưa được định nghĩa trong PIN_MAPPING")
                return
            self.checkInput21 = False
            threading.Thread(
                target=self.open_threading,
                args=(io_pin, self.setStatus21, duration),
                daemon=True
            ).start()

    def setStatus3(self):
        self.checkInput3 = True

    def setStatus5(self):
        self.checkInput5 = True

    def setStatus19(self):
        self.checkInput19 = True

    def setStatus21(self):
        self.checkInput21 = True

    def open_threading(self, io_pin, setStatus, duration=0.5):
        """
        Thread function để điều khiển GPIO (gpiod v2.x)

        Args:
            io_pin: Số physical pin
            setStatus: Callback function để reset trạng thái
            duration: Thời gian mở cửa (giây), mặc định 0.5s
        """
        try:
            line_info = self._get_line(io_pin)
            line_request = line_info['request']
            line_num = line_info['line_num']

            # Turn on (LOW = active)
            line_request.set_value(line_num, Value.INACTIVE)
            print(f"✓ Physical pin {io_pin} → LOW (active)")
            time.sleep(duration)

            # Turn off (HIGH = inactive)
            line_request.set_value(line_num, Value.ACTIVE)
            print(f"✓ Physical pin {io_pin} → HIGH (inactive)")

            # Reset status
            setStatus()

        except Exception as e:
            print(f"❌ Lỗi khi điều khiển physical pin {io_pin}: {e}")
            setStatus()

    def cleanup(self):
        """Giải phóng tài nguyên GPIO (gpiod v2.x)"""
        try:
            # Release tất cả line requests
            for pin, line_info in self.lines.items():
                try:
                    line_info['request'].release()
                    print(f"✓ Đã release physical pin {pin}")
                except:
                    pass

            # Đóng tất cả chips
            for chip_num, chip in self.chips.items():
                try:
                    chip.close()
                    print(f"✓ Đã đóng gpiochip{chip_num}")
                except:
                    pass

            self.lines.clear()
            self.chips.clear()

        except Exception as e:
            print(f"⚠️  Lỗi khi cleanup: {e}")
gpi_radxa_barrie = GPIOBarrieRadxa()
