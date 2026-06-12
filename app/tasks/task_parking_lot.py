import asyncio
import datetime
import os
import time
import traceback
from queue import Empty, Full, Queue

import httpx

from app.utils.orangepi_gpio import gpio_barrie_orangepi
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.models.parking_lot_event import ParkingLotEvent
from app.services.parking_lot_service import parking_lot_service
from app.services.identity_plate_service import identity_plate_service

# Project-root /uploads — same directory mounted as static in main.py.
UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class TaskParkingLot:
    TIME_EXPIRED = 20  # thời gian lưu trữ thông tin (giây)

    def __init__(self):
        # Bounded: every detection enqueues here, so a stalled consumer (slow
        # DB / BLE endpoint) must shed load instead of growing without limit.
        # Entries older than TIME_EXPIRED are useless for correlation anyway.
        self.task_queue = Queue(maxsize=256)
        self.faces = {}
        self.plates = {}
        # Strong refs to in-flight fire-and-forget BLE notifications so the
        # event loop doesn't garbage-collect them mid-request.
        self._bg_tasks = set()
        # Built inside the worker's own event loop (see _run); the queue
        # consumer runs in a dedicated thread so it can't share the global
        # async engine (asyncpg ties connections to the loop that made them).
        self._session_factory = None

    async def valid_success(self, lot, identity_id, plate_number, face_jpeg, plate_jpeg):
        lot_id = lot["id"] if lot else None
        print(f"Valid success for parking lot {lot_id} (identity {identity_id}, plate {plate_number})")
        # Save the two full-frame snapshots off the loop, persist, then open.
        face_url = await asyncio.to_thread(
            self._save_image_blocking, face_jpeg, lot_id, "face",
        )
        plate_url = await asyncio.to_thread(
            self._save_image_blocking, plate_jpeg, lot_id, "plate",
        )
        await self._persist_event(
            lot, identity_id, plate_number, face_url, plate_url,
        )
        await asyncio.to_thread(self._open_barrier, lot_id)

    @staticmethod
    def _save_image_blocking(jpeg, lot_id, kind):
        if not jpeg:
            return None
        try:
            date = datetime.date.today().isoformat()
            folder_rel = os.path.join(
                "parking", str(lot_id if lot_id is not None else "unknown"), date,
            )
            folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
            os.makedirs(folder_abs, exist_ok=True)
            stem = f"{int(time.time() * 1000)}_{kind}"
            with open(os.path.join(folder_abs, f"{stem}.jpg"), "wb") as fp:
                fp.write(jpeg)
            return f"/uploads/{folder_rel}/{stem}.jpg"
        except Exception as e:
            print(f"parking image save error: {e}")
            return None

    def _open_barrier(self, lot_id):
        try:
            # gpio_barrie_orangepi.open_barrie(5)
            # response.raise_for_status()
            print(f"Barrier opened for parking lot {lot_id}")
        except Exception as e:
            print(f"Failed to open barrier for parking lot {lot_id}: {e}")

    async def _persist_event(
        self, lot, identity_id, plate_number, face_url, plate_url,
    ):
        if self._session_factory is None:
            return
        try:
            async with self._session_factory() as db:
                event = ParkingLotEvent(
                    parking_lot_id=lot["id"] if lot else None,
                    identity_id=identity_id,
                    plate_number=plate_number,
                    face_camera_id=lot["face_camera_id"] if lot else None,
                    plate_camera_id=lot["plate_camera_id"] if lot else None,
                    face_image_full=face_url,
                    plate_image_full=plate_url,
                    timestamp=int(time.time()),
                )
                db.add(event)
                await db.commit()
        except Exception as e:
            print(f"parking lot event persist error: {e}")

    async def _notify_ble_detect(self, mac, identity_id, camera_id):
        # Best-effort call to the Bluetooth service. Failures must not block
        # barrier logic, so we swallow every error and just log it.
        if not settings.BLE_DETECT_URL:
            return
        payload = {
            "mac": mac,
            "identity_id": identity_id,
            "camera_id": str(camera_id) if camera_id is not None else "",
        }
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Skip /detect if the service is already scanning this MAC.
                scan_resp = await client.get(
                    settings.BLE_DETECT_URL + "/is-scanning",
                    params={"mac": mac},
                )
                scan_resp.raise_for_status()
                if scan_resp.json().get("scanning"):
                    print(f"BLE already scanning {mac}, skip /detect")
                    return

                response = await client.post(
                    settings.BLE_DETECT_URL + "/detect", json=payload,
                )
                response.raise_for_status()
                print(f"BLE detect notified: {payload} -> {response.status_code}")
        except Exception as e:
            print(f"BLE detect notify failed for {payload}: {e}")

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
            full_jpeg = task.get("full_jpeg")
            key = f"{identity_id}_{camera_id}"
            _camera_plate = parking_lot_service.get_plate_camera(camera_id)

            if not _camera_plate:
                return

            if key in self.faces:
                # Sliding window: a continuously-detected face keeps its entry
                # alive (and its snapshot fresh) instead of expiring 20s after
                # the FIRST sighting. The partner plate branch handles the
                # actual correlation when the plate arrives.
                self.faces[key]["timestamp"] = timestamp
                self.faces[key]["full_jpeg"] = full_jpeg
                return

            lot = parking_lot_service.get_by_camera_id(camera_id)
            _plates = identity_plate_service.get_by_identity_id(identity_id)
            for plate_number in _plates:
                plate_key = f"{plate_number}_{_camera_plate}"
                if plate_key in self.plates:
                    await self.valid_success(
                        lot, identity_id, plate_number,
                        full_jpeg, self.plates[plate_key].get("full_jpeg"),
                    )
                    break
            print(f"Registering face {identity_id} from camera {camera_id}")
            self.faces[key] = {
                "timestamp": timestamp,
                "camera_id": camera_id,
                "full_jpeg": full_jpeg,
            }

        elif name_task == "plate_recognition":
            plate = task.get("plate")
            timestamp = task.get("timestamp")
            camera_id = task.get("camera_id")
            full_jpeg = task.get("full_jpeg")
            _camera_face = parking_lot_service.get_face_camera(camera_id)
            # Fuzzy match so a 1-2 char OCR slip still maps to the right
            # registered plate instead of dropping the event. The same plate can
            # belong to several people, so this returns every matching identity.
            _candidates = identity_plate_service.get_by_plate_fuzzy(plate)

            if not _camera_face or not _candidates:
                return

            # All candidates share the same normalised plate number (alnum,
            # upper) so the key matches the one the face branch builds from
            # get_by_identity_id.
            plate_number = _candidates[0]["plate_number"]
            key = f"{plate_number}_{camera_id}"

            if key in self.plates:
                # Sliding window: refresh instead of expiring 20s after the
                # first read. Skip the duplicate work (BLE + barrier) below.
                self.plates[key]["timestamp"] = timestamp
                self.plates[key]["full_jpeg"] = full_jpeg
                return

            # Notify the BLE service for every co-owner of the plate, but
            # fire-and-forget: a slow/blocking call here would stall the single
            # worker loop and could push later events past the 20s window.
            for _data in _candidates:
                _mac = _data.get("mac_bluetooth")
                if _mac and settings.BLE_DETECT_URL:
                    bg = asyncio.create_task(
                        self._notify_ble_detect(
                            _mac, _data["identity_id"], _camera_face,
                        )
                    )
                    self._bg_tasks.add(bg)
                    bg.add_done_callback(self._bg_tasks.discard)

            lot = parking_lot_service.get_by_camera_id(camera_id)
            # Open the barrier for whichever co-owner of the plate has a recent
            # matching face — stop at the first hit.
            for _data in _candidates:
                _identity_id = _data["identity_id"]
                _face_key = f"{_identity_id}_{_camera_face}"
                if _face_key in self.faces:
                    print(
                        f"[parking_lot] match identity={_identity_id} "
                        f"plate={plate_number} "
                        f"mac_bluetooth={_data.get('mac_bluetooth')}"
                    )
                    await self.valid_success(
                        lot, _identity_id, plate_number,
                        self.faces[_face_key].get("full_jpeg"), full_jpeg,
                    )
                    break
            print(f"Registering plate {plate_number} from camera {camera_id}")
            self.plates[key] = {
                "timestamp": timestamp,
                "camera_id": camera_id,
                "full_jpeg": full_jpeg,
            }

    def add_task(self, task):
        # Never block the AI recv loop: when the queue is full, drop the
        # oldest entry — it is the least likely to still fall inside the
        # TIME_EXPIRED correlation window.
        while True:
            try:
                self.task_queue.put_nowait(task)
                return
            except Full:
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except Empty:
                    pass


task_parking_lot = TaskParkingLot()
