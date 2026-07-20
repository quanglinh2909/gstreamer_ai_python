import threading
import time

try:
    import pyfirmata
    import serial.tools.list_ports
except ImportError:
    pyfirmata = None
    serial = None


class ArduinoBarrie:

    def __init__(self):
        self.checkFirst = False
        self.listInput = []
        # Set mode GPIO

    def open_barrie(self, input, duration=0.5):
        if input in self.listInput:
            return
        self.listInput.append(input)
        t = threading.Thread(target=self.open_barrie_thead, args=(input, duration), daemon=True)
        t.start()


    def open_barrie_thead(self, input, duration=0.5):

        try:
            if self.checkFirst == False:
                self.board = pyfirmata.Arduino(self.get_port_arduino())
                self.checkFirst = True
            self.board.digital[input].write(1)
            time.sleep(duration)
            self.board.digital[input].write(0)
        except:
            try:
                self.board = pyfirmata.Arduino(self.get_port_arduino())
                self.board.digital[input].write(1)
                time.sleep(duration)
                self.board.digital[input].write(0)
            except:
                pass
        finally:
            try:
                self.listInput.remove(input)
            except Exception as e:
                print(f"Error removing input {input} from list: {e}")


    def get_port_arduino(self):
        try:
            ports = serial.tools.list_ports.comports()
            portResult = None
            for port, desc, hwid in sorted(ports):
                if 'USB Serial' in desc:
                    portResult = port
                    break
            return portResult
        except:
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.checkFirst and hasattr(self, 'board'):
                self.board.exit()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.checkFirst = False
            self.listInput.clear()

arduino_barrie = ArduinoBarrie()