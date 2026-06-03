import re
import threading
from typing import Dict, List, Optional

import Levenshtein
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.identity_plate import IdentityPlate
from app.repositories.identity_plate_repository import IdentityPlateRepository
from app.repositories.identity_repository import IdentityRepository


def _normalize(plate_number: str) -> str:
    # Keep only letters and digits, upper-cased — drop spaces, dots, dashes
    # and every other symbol so the cache key matches the ALPR plate string
    # regardless of how the plate was punctuated on input.
    return re.sub(r"[^A-Za-z0-9]", "", plate_number or "").upper()


def _to_dict(entry: IdentityPlate) -> Dict:
    return {
        "id": entry.id,
        "identity_id": entry.identity_id,
        "plate_number": entry.plate_number,
    }


class IdentityPlateService:
    def __init__(self):
        # plate_number (normalised, UPPER) -> {id, identity_id, plate_number}.
        # Mirrors plate_white_list in PlateWhiteListService: an in-memory copy
        # so the ALPR pipeline can map a detected plate to its identity without
        # hitting the DB on every event. Lookups stay O(1).
        self.identity_plates: Dict[str, Dict] = {}
        # Cross-thread access: API handlers (FastAPI loop) mutate; the
        # process_ai_service thread reads on every detection.
        self._cache_lock = threading.RLock()

    async def load_all(self, db: AsyncSession) -> None:
        """Prime the in-memory cache from DB. Call once at app startup
        (after Base.metadata.create_all) so the first lookup doesn't have
        to wait on I/O."""
        rows = await IdentityPlateRepository.list_all(db)
        with self._cache_lock:
            self.identity_plates = {
                _normalize(row.plate_number): _to_dict(row) for row in rows
            }
        print(f"[identity plate] loaded {len(self.identity_plates)} entries")

    def get_by_plate(self, plate_number: str) -> Optional[Dict]:
        """Thread-safe O(1) lookup of the identity a plate belongs to."""
        normalized = _normalize(plate_number)
        with self._cache_lock:
            return self.identity_plates.get(normalized)

    def get_by_plate_fuzzy(
        self, plate_number: str, max_distance: int = 2,
    ) -> Optional[Dict]:
        """Like get_by_plate but tolerant of OCR noise: tries an exact match
        first, then falls back to the closest registered plate within
        `max_distance` edits (Levenshtein). Returns None when nothing is close
        enough. To avoid false hits on short plates, the allowed distance is
        capped at len // 4 (so a 4-char plate allows 0, an 8-char allows 2)."""
        normalized = _normalize(plate_number)
        if not normalized:
            return None
        # Shorter plates get a tighter budget — 2 edits on a 4-char plate could
        # match a completely different vehicle.
        budget = min(max_distance, len(normalized) // 4)
        with self._cache_lock:
            exact = self.identity_plates.get(normalized)
            if exact is not None:
                return exact
            best, best_dist = None, budget + 1
            for key, data in self.identity_plates.items():
                dist = Levenshtein.distance(normalized, key)
                if dist < best_dist:
                    best, best_dist = data, dist
            return best if best_dist <= budget else None

    def get_by_identity_id(self, identity_id: int) -> List[str]:
        """Thread-safe lookup of every cached plate number for an identity,
        e.g. ["51F-123", "51F-124"]. Returns an empty list when the identity
        has no plates."""
        with self._cache_lock:
            return [
                plate["plate_number"]
                for plate in self.identity_plates.values()
                if plate["identity_id"] == identity_id
            ]

    def all_cached(self) -> List[Dict]:
        with self._cache_lock:
            return list(self.identity_plates.values())

    async def _require_identity(self, db: AsyncSession, identity_id: int) -> None:
        identity = await IdentityRepository.get(db, identity_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="Identity not found")

    async def list_by_identity(
        self, db: AsyncSession, identity_id: int,
    ) -> List[IdentityPlate]:
        await self._require_identity(db, identity_id)
        return await IdentityPlateRepository.list_by_identity(db, identity_id)

    async def create(
        self,
        db: AsyncSession,
        identity_id: int,
        plate_number: str,
    ) -> IdentityPlate:
        await self._require_identity(db, identity_id)
        normalized = _normalize(plate_number)
        if not normalized:
            raise HTTPException(status_code=400, detail="Plate number is empty")
        existing = await IdentityPlateRepository.get_by_identity_and_plate(
            db, identity_id, normalized,
        )
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Plate {normalized} already assigned to this identity",
            )
        entry = await IdentityPlateRepository.create(db, identity_id, normalized)
        with self._cache_lock:
            self.identity_plates[_normalize(entry.plate_number)] = _to_dict(entry)
        return entry

    async def get(self, db: AsyncSession, entry_id: int) -> IdentityPlate:
        entry = await IdentityPlateRepository.get(db, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Plate not found")
        return entry

    async def update(
        self,
        db: AsyncSession,
        entry_id: int,
        plate_number: str,
    ) -> IdentityPlate:
        entry = await self.get(db, entry_id)
        normalized = _normalize(plate_number)
        if not normalized:
            raise HTTPException(status_code=400, detail="Plate number is empty")
        # Only check uniqueness when the value actually changes — otherwise a
        # self-update would falsely conflict with its own row.
        if normalized.upper() != entry.plate_number.upper():
            existing = await IdentityPlateRepository.get_by_identity_and_plate(
                db, entry.identity_id, normalized,
            )
            if existing is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Plate {normalized} already assigned to this identity",
                )
        old_key = _normalize(entry.plate_number)
        entry = await IdentityPlateRepository.update(db, entry, normalized)
        new_key = _normalize(entry.plate_number)
        with self._cache_lock:
            # Drop the old key first in case the plate number changed, otherwise
            # the rename would leave a stale duplicate in the cache.
            if old_key != new_key:
                self.identity_plates.pop(old_key, None)
            self.identity_plates[new_key] = _to_dict(entry)
        return entry

    async def delete(self, db: AsyncSession, entry_id: int) -> None:
        entry = await self.get(db, entry_id)
        # capture before SQLAlchemy invalidates the row
        key = _normalize(entry.plate_number)
        await IdentityPlateRepository.delete(db, entry)
        with self._cache_lock:
            self.identity_plates.pop(key, None)


identity_plate_service = IdentityPlateService()
