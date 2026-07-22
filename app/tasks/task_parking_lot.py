import asyncio
import datetime
import os
import time
import traceback
from queue import Empty, Full, Queue

import httpx

from app.utils.open_door.door_manager import door_manager
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.models.parking_lot_event import ParkingLotEvent
from app.services.parking_lot_service import parking_lot_service
from app.services.identity_plate_service import identity_plate_service
from app.utils.plate_recognition_hepper import detect_plate_from_children

# Project-root /uploads — same directory mounted as static in main.py.
UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class TaskParkingLot:
    # Mọi ngưỡng (cửa sổ ghép cặp, cooldown, độ dài xung barrier, sai số ký
    # tự, ngưỡng OCR) nay nằm ở từng dòng ParkingLot — xem models/parking_lot
    # .py. Cache của parking_lot_service đã có sẵn trong RAM nên đọc chúng
    # trên đường nóng không phát sinh truy vấn DB nào.

    def __init__(self):
        # Bounded: every detection enqueues here, so a stalled consumer (slow
        # DB / BLE endpoint) must shed load instead of growing without limit.
        # Entries past their expiry are useless for correlation anyway.
        self.task_queue = Queue(maxsize=256)
        # Ba dict dưới đây DÙNG CHUNG cho mọi bãi, trong khi mỗi bãi có cửa sổ
        # riêng — nên hạn được tính sẵn lúc chèn và lưu ngay trong entry
        # ("expire_at"). Vòng quét không cần biết entry thuộc bãi nào, chỉ so
        # với now. Nếu quét bằng một hằng số chung thì bãi có cửa sổ ngắn sẽ
        # giữ entry quá lâu và ngược lại.
        self.faces = {}
        self.plates = {}
        # f"{plate_number}_{plate_camera_id}" -> thời điểm HẾT HẠN cooldown.
        # Tách khỏi self.plates vì entry trong self.plates có thể hết hạn rồi
        # được tạo lại NGAY trong cùng một lượt xe — cờ matched gắn trên entry
        # sẽ mất theo, còn dict này thì sống hết cooldown.
        self.matches = {}
        # Strong refs to in-flight fire-and-forget BLE notifications so the
        # event loop doesn't garbage-collect them mid-request.
        self._bg_tasks = set()
        # Built inside the worker's own event loop (see _run); the queue
        # consumer runs in a dedicated thread so it can't share the global
        # async engine (asyncpg ties connections to the loop that made them).
        self._session_factory = None

    async def valid_success(self, lot, identity_id, plate_number, face_jpeg, plate_jpeg):
        lot_id = lot["id"] if lot else None
        duration = float(lot["barrier_duration"])
        print(f"Valid success for parking lot {lot_id} (identity {identity_id}, plate {plate_number})")
        # Mở barrier TRƯỚC — đây là thứ người ở cổng đang chờ. Trước kia nó
        # đứng CUỐI, sau 2 lần ghi ảnh + 1 lần ghi DB: bình thường chỉ tốn
        # vài chục ms, nhưng khi DB chậm/rớt thì barrier phải đợi hết timeout
        # của asyncpg. (open_door tự tách thread cho xung GPIO nên lệnh này
        # trả về ngay.)
        await asyncio.to_thread(self._open_barrier, lot_id, duration)
        # Lưu ảnh + ghi DB chạy NỀN để worker quay lại ghép cặp ngay lập tức
        # — DB nghẽn không giữ chân barrier của xe kế tiếp / làn khác nữa.
        bg = asyncio.create_task(
            self._persist_success(lot, identity_id, plate_number, face_jpeg, plate_jpeg)
        )
        self._bg_tasks.add(bg)
        bg.add_done_callback(self._bg_tasks.discard)

    async def _persist_success(self, lot, identity_id, plate_number, face_jpeg, plate_jpeg):
        # Background half of valid_success: snapshots to disk, then the DB row.
        lot_id = lot["id"] if lot else None
        face_url = await asyncio.to_thread(
            self._save_image_blocking, face_jpeg, lot_id, "face",
        )
        plate_url = await asyncio.to_thread(
            self._save_image_blocking, plate_jpeg, lot_id, "plate",
        )
        await self._persist_event(
            lot, identity_id, plate_number, face_url, plate_url,
        )

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

    def _open_barrier(self, lot_id, duration):
        try:
            door_manager.open_door(duration)
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

    def _claim_match(self, plate_key, now, cooldown):
        """True nếu biển này (ở làn này) CHƯA tạo sự kiện trong cooldown —
        đồng thời ghi nhận luôn. False = lượt xe này đã có sự kiện rồi
        (ví dụ người thứ 2 ngồi cùng xe) -> bỏ qua, không tạo dòng mới."""
        if now < self.matches.get(plate_key, 0):
            return False
        self.matches[plate_key] = now + cooldown
        return True

    async def _handle(self, task):
        # Drop entries past the expiry stamped on them when they were added.
        now = time.time()
        self.faces = {
            k: v for k, v in self.faces.items() if now < v["expire_at"]
        }
        self.plates = {
            k: v for k, v in self.plates.items() if now < v["expire_at"]
        }
        self.matches = {
            k: expire_at for k, expire_at in self.matches.items() if now < expire_at
        }

        name_task = task.get("task")
        if name_task == "face_recognition":
            identity_id = task.get("identity_id")
            timestamp = task.get("timestamp")
            camera_id = task.get("camera_id")
            full_jpeg = task.get("full_jpeg")
            key = f"{identity_id}_{camera_id}"
            _camera_plate = parking_lot_service.get_plate_camera(camera_id)
            lot = parking_lot_service.get_by_camera_id(camera_id)

            # lot lấy sớm hơn trước (cũ chỉ lấy sau nhánh làm mới) vì cửa sổ
            # và cooldown giờ đọc từ chính nó.
            if not _camera_plate or lot is None:
                return

            window = lot["time_expired"]
            cooldown = lot["match_cooldown"]

            if key in self.faces:
                # Sliding window: a continuously-detected face keeps its entry
                # alive (and its snapshot fresh) instead of expiring `window`
                # seconds after the FIRST sighting. The partner plate branch
                # handles the actual correlation when the plate arrives.
                self.faces[key]["timestamp"] = timestamp
                self.faces[key]["expire_at"] = now + window
                self.faces[key]["full_jpeg"] = full_jpeg
                # Người vẫn đứng ở làn -> trượt luôn cửa sổ chống trùng của
                # các biển liên quan, phòng ca camera biển bị che quá lâu rồi
                # đọc lại biển trong cùng lượt xe.
                for _plate_number in identity_plate_service.get_by_identity_id(identity_id):
                    _pk = f"{_plate_number}_{_camera_plate}"
                    if _pk in self.matches:
                        self.matches[_pk] = now + cooldown
                return

            _plates = identity_plate_service.get_by_identity_id(identity_id)
            for plate_number in _plates:
                plate_key = f"{plate_number}_{_camera_plate}"
                if plate_key in self.plates:
                    if self._claim_match(plate_key, now, cooldown):
                        await self.valid_success(
                            lot, identity_id, plate_number,
                            full_jpeg, self.plates[plate_key].get("full_jpeg"),
                        )
                    else:
                        # 2 người cùng xe: biển này vừa tạo sự kiện với khuôn
                        # mặt khác rồi — không tạo dòng thứ 2 / mở cửa lần 2.
                        print(
                            f"[parking_lot] plate {plate_number} đã có sự kiện "
                            f"trong lượt này, bỏ qua face {identity_id}"
                        )
                    break
            print(f"Registering face {identity_id} from camera {camera_id}")
            self.faces[key] = {
                "timestamp": timestamp,
                "expire_at": now + window,
                "camera_id": camera_id,
                "full_jpeg": full_jpeg,
            }

        elif name_task == "plate_recognition":
            timestamp = task.get("timestamp")
            camera_id = task.get("camera_id")
            full_jpeg = task.get("full_jpeg")
            _camera_face = parking_lot_service.get_face_camera(camera_id)
            lot = parking_lot_service.get_by_camera_id(camera_id)

            if not _camera_face or lot is None:
                return

            window = lot["time_expired"]
            cooldown = lot["match_cooldown"]

            # Bãi xe TỰ đọc lại biển từ các OCR children bằng ocr_confidence
            # của chính nó, không dùng lại chuỗi mà plate_recognition_service
            # đã dựng bằng secondaryConf của AI job — siết ngưỡng cho barrier
            # không kéo theo thay đổi dữ liệu EventPlate, và ngược lại.
            plate = detect_plate_from_children(
                task.get("children") or [], lot["ocr_confidence"],
            )
            if not plate:
                return

            # Fuzzy match so a 1-2 char OCR slip still maps to the right
            # registered plate instead of dropping the event. The same plate can
            # belong to several people, so this returns every matching identity.
            _candidates = identity_plate_service.get_by_plate_fuzzy(
                plate, lot["max_edit_distance"],
            )
            if not _candidates:
                return

            # All candidates share the same normalised plate number (alnum,
            # upper) so the key matches the one the face branch builds from
            # get_by_identity_id.
            plate_number = _candidates[0]["plate_number"]
            key = f"{plate_number}_{camera_id}"

            if key in self.plates:
                # Sliding window: refresh instead of expiring `window` seconds
                # after the first read. Skip the duplicate work (BLE + barrier)
                # below.
                self.plates[key]["timestamp"] = timestamp
                self.plates[key]["expire_at"] = now + window
                self.plates[key]["full_jpeg"] = full_jpeg
                # Xe còn đứng ở làn (biển vẫn đang được đọc) -> trượt luôn
                # cửa sổ chống trùng, để mặt người thứ 2 nhận diện muộn vẫn
                # không tạo thêm sự kiện cho cùng lượt xe.
                if key in self.matches:
                    self.matches[key] = now + cooldown
                return

            # Notify the BLE service for every co-owner of the plate, but
            # fire-and-forget: a slow/blocking call here would stall the single
            # worker loop and could push later events past the correlation
            # window.
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

            # Open the barrier for whichever co-owner of the plate has a recent
            # matching face — stop at the first hit. _claim_match chặn lượt xe
            # đã có sự kiện (entry biển hết hạn rồi được đọc lại trong cùng
            # lượt sẽ đi qua nhánh này lần nữa).
            for _data in _candidates:
                _identity_id = _data["identity_id"]
                _face_key = f"{_identity_id}_{_camera_face}"
                if _face_key in self.faces:
                    if self._claim_match(key, now, cooldown):
                        print(
                            f"[parking_lot] match identity={_identity_id} "
                            f"plate={plate_number} "
                            f"mac_bluetooth={_data.get('mac_bluetooth')}"
                        )
                        await self.valid_success(
                            lot, _identity_id, plate_number,
                            self.faces[_face_key].get("full_jpeg"), full_jpeg,
                        )
                    else:
                        print(
                            f"[parking_lot] plate {plate_number} đã có sự kiện "
                            f"trong lượt này, bỏ qua"
                        )
                    break
            print(f"Registering plate {plate_number} from camera {camera_id}")
            self.plates[key] = {
                "timestamp": timestamp,
                "expire_at": now + window,
                "camera_id": camera_id,
                "full_jpeg": full_jpeg,
            }

    def add_task(self, task):
        # Never block the AI recv loop: when the queue is full, drop the
        # oldest entry — it is the least likely to still fall inside the
        # correlation window.
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
