import asyncio
import time
import traceback
from queue import Queue

import requests
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.models.parking_lot_event import ParkingLotEvent
from app.services.parking_lot_service import parking_lot_service
from app.services.identity_plate_service import identity_plate_service


class TaskParkingLot:
    TIME_EXPIRED = 20  # thời gian lưu trữ thông tin (giây)

    def __init__(self):
        self.task_queue = Queue()
        self.faces = {}
        self.plates = {}
        # Built inside the worker's own event loop (see _run); the queue
        # consumer runs in a dedicated thread so it can't share the global
        # async engine (asyncpg ties connections to the loop that made them).
        self._session_factory = None

    async def valid_success(self, camera_id, identity_id, plate_number):
        print(f"Valid success for camera {camera_id}")
        # Persist the gate-open event, then physically open the barrier.
        await self._persist_event(camera_id, identity_id, plate_number)
        await asyncio.to_thread(self._open_barrier, camera_id)

    def _open_barrier(self, camera_id):
        data_send = {"io_pin": 5}
        try:
            requests.post(
                "http://localhost:8087/barrier/open", json=data_send, timeout=5.0,
            )
            # response.raise_for_status()
            print(f"Barrier opened for camera {camera_id}")
        except Exception as e:
            print(f"Failed to open barrier for camera {camera_id}: {e}")

    async def _persist_event(self, camera_id, identity_id, plate_number):
        if self._session_factory is None:
            return
        # camera_id is whichever side triggered the match; the lot carries both
        # paired cameras so we store the full context regardless.
        lot = parking_lot_service.get_by_camera_id(camera_id)
        try:
            async with self._session_factory() as db:
                event = ParkingLotEvent(
                    parking_lot_id=lot["id"] if lot else None,
                    identity_id=identity_id,
                    plate_number=plate_number,
                    face_camera_id=lot["face_camera_id"] if lot else None,
                    plate_camera_id=lot["plate_camera_id"] if lot else None,
                    timestamp=int(time.time()),
                )
                db.add(event)
                await db.commit()
        except Exception as e:
            print(f"parking lot event persist error: {e}")

    def worker(self):
        # Thread entrypoint: run the async consumer on its own event loop.
        try:
            asyncio.run(self._run())
        except Exception as exc:
            print(f"task_parking_lot worker crashed: {exc}")
            traceback.print_exc()

    async def _run(self):
        engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
        self._session_factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        try:
            while True:
                # Block for the next task off the event loop, then process it.
                task = await asyncio.to_thread(self.task_queue.get)
                try:
                    if task is not None:
                        await self._handle(task)
                except Exception as e:
                    print(f"Error processing task: {e}")
                    traceback.print_exc()
                finally:
                    self.task_queue.task_done()
        finally:
            await engine.dispose()

    async def _handle(self, task):
        # Drop entries older than the correlation window.
        now = time.time()
        self.faces = {
            k: v for k, v in self.faces.items()
            if now - v["timestamp"] < self.TIME_EXPIRED
        }
        self.plates = {
            k: v for k, v in self.plates.items()
            if now - v["timestamp"] < self.TIME_EXPIRED
        }

        name_task = task.get("task")
        if name_task == "face_recognition":
            identity_id = task.get("identity_id")
            timestamp = task.get("timestamp")
            camera_id = task.get("camera_id")
            key = f"{identity_id}_{camera_id}"
            _camera_plate = parking_lot_service.get_plate_camera(camera_id)

            if key in self.faces or not _camera_plate:
                print(f"Face {identity_id} from camera {camera_id} already processed or no plate camera linked")
                return

            _plates = identity_plate_service.get_by_identity_id(identity_id)
            for plate_number in _plates:
                plate_key = f"{plate_number}_{_camera_plate}"
                if plate_key in self.plates:
                    await self.valid_success(camera_id, identity_id, plate_number)
                    break
            print(f"Registering face {identity_id} from camera {camera_id}")
            self.faces[key] = {"timestamp": timestamp, "camera_id": camera_id}

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
                return

            # Use the normalised plate number (alnum, upper) so the key matches
            # the one the face branch builds from get_by_identity_id.
            plate_number = _data["plate_number"]
            key = f"{plate_number}_{camera_id}"
            if key in self.plates:
                print(f"Plate {plate_number} from camera {camera_id} already processed")
                return

            _face_key = f"{_identity_id}_{_camera_face}"
            if _face_key in self.faces:
                await self.valid_success(camera_id, _identity_id, plate_number)
            print(f"Registering plate {plate_number} from camera {camera_id}")
            self.plates[key] = {"timestamp": timestamp, "camera_id": camera_id}

    def add_task(self, task):
        self.task_queue.put(task)


task_parking_lot = TaskParkingLot()
