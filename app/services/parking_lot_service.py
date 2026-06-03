import threading
from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parking_lot import ParkingLot
from app.repositories.parking_lot_repository import ParkingLotRepository


def _clean_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    trimmed = name.strip()
    return trimmed or None


def _to_dict(entry: ParkingLot) -> Dict:
    return {
        "id": entry.id,
        "name": entry.name,
        "face_camera_id": entry.face_camera_id,
        "plate_camera_id": entry.plate_camera_id,
    }


class ParkingLotService:
    def __init__(self):
        # id -> parking lot dict. Mirrors plate_white_list in
        # PlateWhiteListService: an in-memory copy so the AI pipeline can
        # correlate the face/plate cameras of a gate without hitting the DB
        # on every event.
        self.parking_lots: Dict[int, Dict] = {}
        # camera_id (face OR plate) -> parking lot dict, for O(1) lookup of
        # "which lot — and which paired camera — does this stream belong to".
        self._by_camera: Dict[str, Dict] = {}
        # Cross-thread access: API handlers (FastAPI loop) mutate; the
        # process_ai_service thread reads.
        self._cache_lock = threading.RLock()

    async def load_all(self, db: AsyncSession) -> None:
        """Prime the in-memory cache from DB. Call once at app startup
        (after Base.metadata.create_all) so the first lookup doesn't have
        to wait on I/O."""
        rows = await ParkingLotRepository.list_all(db)
        with self._cache_lock:
            self.parking_lots = {row.id: _to_dict(row) for row in rows}
            self._by_camera = {}
            for row in rows:
                self._by_camera[row.face_camera_id] = self.parking_lots[row.id]
                self._by_camera[row.plate_camera_id] = self.parking_lots[row.id]
        print(f"[parking lot] loaded {len(self.parking_lots)} entries")

    def _cache_put(self, entry: ParkingLot) -> None:
        data = _to_dict(entry)
        with self._cache_lock:
            # Drop any stale camera keys this lot used to own before re-adding,
            # otherwise a camera swap would leave a dangling pointer.
            self._cache_drop(entry.id)
            self.parking_lots[entry.id] = data
            self._by_camera[entry.face_camera_id] = data
            self._by_camera[entry.plate_camera_id] = data

    def _cache_drop(self, entry_id: int) -> None:
        with self._cache_lock:
            old = self.parking_lots.pop(entry_id, None)
            if old is not None:
                self._by_camera.pop(old["face_camera_id"], None)
                self._by_camera.pop(old["plate_camera_id"], None)

    def get_by_camera_id(self, camera_id: str) -> Optional[Dict]:
        """Thread-safe O(1) lookup of the parking lot a camera belongs to."""
        with self._cache_lock:
            return self._by_camera.get(camera_id)

    def get_plate_camera(self, face_camera_id: str) -> Optional[str]:
        """Given a face camera, return the plate camera paired with it in the
        same parking lot. None if the id isn't a registered face camera."""
        with self._cache_lock:
            lot = self._by_camera.get(face_camera_id)
            if lot is None or lot["face_camera_id"] != face_camera_id:
                return None
            return lot["plate_camera_id"]

    def get_face_camera(self, plate_camera_id: str) -> Optional[str]:
        """Given a plate camera, return the face camera paired with it in the
        same parking lot. None if the id isn't a registered plate camera."""
        with self._cache_lock:
            lot = self._by_camera.get(plate_camera_id)
            if lot is None or lot["plate_camera_id"] != plate_camera_id:
                return None
            return lot["face_camera_id"]

    def all_cached(self) -> List[Dict]:
        with self._cache_lock:
            return list(self.parking_lots.values())

    async def _check_cameras_free(
        self,
        db: AsyncSession,
        face_camera_id: str,
        plate_camera_id: str,
        exclude_id: Optional[int] = None,
    ) -> None:
        if face_camera_id == plate_camera_id:
            raise HTTPException(
                status_code=400,
                detail="Face camera and plate camera must be different",
            )
        for camera_id in (face_camera_id, plate_camera_id):
            existing = await ParkingLotRepository.get_by_camera(db, camera_id)
            if existing is not None and existing.id != exclude_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"Camera {camera_id} already linked to a parking lot",
                )

    async def create(
        self,
        db: AsyncSession,
        face_camera_id: str,
        plate_camera_id: str,
        name: Optional[str] = None,
    ) -> ParkingLot:
        await self._check_cameras_free(db, face_camera_id, plate_camera_id)
        entry = await ParkingLotRepository.create(
            db, face_camera_id, plate_camera_id, _clean_name(name),
        )
        self._cache_put(entry)
        return entry

    async def get(self, db: AsyncSession, entry_id: int) -> ParkingLot:
        entry = await ParkingLotRepository.get(db, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Parking lot not found")
        return entry

    async def update(
        self,
        db: AsyncSession,
        entry_id: int,
        face_camera_id: str,
        plate_camera_id: str,
        name: Optional[str] = None,
    ) -> ParkingLot:
        entry = await self.get(db, entry_id)
        await self._check_cameras_free(
            db, face_camera_id, plate_camera_id, exclude_id=entry.id,
        )
        entry = await ParkingLotRepository.update(
            db, entry, face_camera_id, plate_camera_id, _clean_name(name),
        )
        self._cache_put(entry)
        return entry

    async def list_paginated(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        name: Optional[str] = None,
    ):
        return await ParkingLotRepository.list_paginated(db, page, size, name)

    async def delete(self, db: AsyncSession, entry_id: int) -> None:
        entry = await self.get(db, entry_id)
        await ParkingLotRepository.delete(db, entry)
        self._cache_drop(entry_id)


parking_lot_service = ParkingLotService()
