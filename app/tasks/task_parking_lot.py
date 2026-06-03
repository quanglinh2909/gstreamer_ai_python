import time
import traceback
from queue import Queue
import requests
from app.services.parking_lot_service import parking_lot_service
from app.services.identity_plate_service import identity_plate_service


class TaskParkingLot:
    TIME_EXPIRED = 20  # thời gian lưu trữ thông tin (giây)
    def __init__(self):
        self.task_queue = Queue()
        self.faces = {}
        self.plates = {}
    
    def valid_success(self, camera_id):
        print(f"Valid success for camera {camera_id}")
        data_send = {"io_pin": 5}
        try:
            response = requests.post("http://localhost:8087/barrier/open", json=data_send, timeout=5.0)
            # response.raise_for_status()
            print(f"Barrier opened for camera {camera_id}")
        except Exception as e:
            print(f"Failed to open barrier for camera {camera_id}: {e}")

    def worker(self):
        while True:
            try:
                task = self.task_queue.get()  # chờ task
                if task is None:  # tín hiệu dừng
                    self.task_queue.task_done()
                    continue
                # remove expired entries
                now = time.time()
                self.faces = {k: v for k, v in self.faces.items() if now - v["timestamp"] < self.TIME_EXPIRED}
                self.plates = {k: v for k, v in self.plates.items() if now - v["timestamp"] < self.TIME_EXPIRED} 

                name_task = task.get("task")
                if name_task == "face_recognition":
                    identity_id = task.get("identity_id")
                    timestamp = task.get("timestamp")
                    camera_id = task.get("camera_id")
                    key = f"{identity_id}_{camera_id}"
                    _camera_plate = parking_lot_service.get_plate_camera(camera_id)

                    if key in self.faces or not _camera_plate:
                        print(f"Face {identity_id} from camera {camera_id} already processed or no plate camera linked")
                        self.task_queue.task_done()
                        continue

                    _plates = identity_plate_service.get_by_identity_id(identity_id)

                    for plate_number in _plates:
                        plate_key = f"{plate_number}_{_camera_plate}"
                        if plate_key in self.plates:
                            self.valid_success(camera_id)
                            break
                    self.faces[key] = {
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                    }

                        
                elif name_task == "plate_recognition":
                    plate = task.get("plate")
                    timestamp = task.get("timestamp")
                    camera_id = task.get("camera_id")
                    _camera_face = parking_lot_service.get_face_camera(camera_id)
                    # Fuzzy match so a 1-2 char OCR slip still maps to the right
                    # registered plate instead of dropping the event.
                    _data = identity_plate_service.get_by_plate_fuzzy(plate)
                    _identity_id = _data.get("identity_id") if _data else None

                    if not _camera_face or not _identity_id:
                        self.task_queue.task_done()
                        continue

                    # Use the normalised plate number (alnum, upper) so the key
                    # matches the one the face branch builds from
                    # get_by_identity_id; the raw OCR string still has spaces.
                    plate_number = _data["plate_number"]
                    key = f"{plate_number}_{camera_id}"
                    if key in self.plates:
                        print(f"Plate {plate_number} from camera {camera_id} already processed")
                        self.task_queue.task_done()
                        continue

                    _face_key = f"{_identity_id}_{_camera_face}"
                    if _face_key in self.faces:
                        self.valid_success(camera_id)

                    self.plates[key] = {
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                    }

                self.task_queue.task_done()
            except Exception as e:
                print(f"Error processing task: {e}")
                traceback.print_exc()
                self.task_queue.task_done()

    def add_task(self, task):
        self.task_queue.put(task)

task_parking_lot = TaskParkingLot()
