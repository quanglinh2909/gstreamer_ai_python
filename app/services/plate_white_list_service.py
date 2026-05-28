import re
import threading
from typing import Dict, List, Optional
import Levenshtein
import httpx
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import time

from app.models.plate_white_list import PlateWhiteList
from app.repositories.plate_white_list_repository import PlateWhiteListRepository


def _normalize(plate_number: str) -> str:
    # Plates are typically compared case- and whitespace-insensitively. Store
    # them canonical (upper-case, no surrounding spaces) so lookups against
    # detected plates from ALPR don't miss because of trivial formatting.
    return plate_number.strip().upper()


def _clean_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    trimmed = name.strip()
    return trimmed or None


class PlateWhiteListService:
    def __init__(self):
        # plate_number (normalised, UPPER) -> {} (placeholder).
        # The ALPR hot path only needs membership ("is this plate
        # whitelisted?"), so the value stays an empty dict — leaving
        # room to attach per-plate metadata later without changing the
        # shape. Lookups stay O(1).
        self.plate_white_list: Dict[str, Dict] = {}
        # Cross-thread access: API handlers (FastAPI loop) mutate; the
        # process_ai_service thread reads on every detection.
        self._cache_lock = threading.RLock()

    async def load_all(self, db: AsyncSession) -> None:
        """Prime the in-memory cache from DB. Call once at app startup
        (after Base.metadata.create_all) so the first lookup doesn't have
        to wait on I/O."""
        rows = await PlateWhiteListRepository.list_all(db)
        with self._cache_lock:
            self.plate_white_list = {row.plate_number: {} for row in rows}
        print(f"[plate whitelist] loaded {len(self.plate_white_list)} entries")

    def is_whitelisted(self, plate_number: str) -> bool:
        """Thread-safe O(1) membership check for the ALPR pipeline."""
        normalized = _normalize(plate_number)
        with self._cache_lock:
            return normalized in self.plate_white_list

    def all_cached(self) -> List[str]:
        with self._cache_lock:
            return list(self.plate_white_list.keys())

    async def create(
        self, db: AsyncSession, plate_number: str, name: Optional[str] = None,
    ) -> PlateWhiteList:
        normalized = _normalize(plate_number)
        if not normalized:
            raise HTTPException(status_code=400, detail="Plate number is empty")
        existing = await PlateWhiteListRepository.get_by_plate(db, normalized)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Plate {normalized} already in whitelist",
            )
        entry = await PlateWhiteListRepository.create(db, normalized, _clean_name(name))
        with self._cache_lock:
            self.plate_white_list[entry.plate_number] = {}
        return entry

    async def update(
        self,
        db: AsyncSession,
        entry_id: int,
        plate_number: str,
        name: Optional[str] = None,
    ) -> PlateWhiteList:
        entry = await PlateWhiteListRepository.get(db, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Whitelist entry not found")
        normalized = _normalize(plate_number)
        if not normalized:
            raise HTTPException(status_code=400, detail="Plate number is empty")
        # Only check uniqueness when the value actually changes — otherwise a
        # self-update would falsely conflict with its own row.
        if normalized != entry.plate_number:
            existing = await PlateWhiteListRepository.get_by_plate(db, normalized)
            if existing is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Plate {normalized} already in whitelist",
                )
        old_plate = entry.plate_number
        entry = await PlateWhiteListRepository.update(
            db, entry, normalized, _clean_name(name),
        )
        with self._cache_lock:
            # Drop the old key first in case the plate number itself changed,
            # otherwise the rename would leave a stale duplicate in the cache.
            if old_plate != entry.plate_number:
                self.plate_white_list.pop(old_plate, None)
            self.plate_white_list[entry.plate_number] = {}
        return entry

    async def get(self, db: AsyncSession, entry_id: int) -> Optional[PlateWhiteList]:
        return await PlateWhiteListRepository.get(db, entry_id)

    async def list_paginated(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        plate_number: Optional[str] = None,
    ):
        return await PlateWhiteListRepository.list_paginated(
            db, page, size, plate_number,
        )

    async def delete(self, db: AsyncSession, entry_id: int) -> None:
        entry = await PlateWhiteListRepository.get(db, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Whitelist entry not found")
        plate = entry.plate_number  # capture before SQLAlchemy invalidates the row
        await PlateWhiteListRepository.delete(db, entry)
        with self._cache_lock:
            self.plate_white_list.pop(plate, None)
    
    async def process_ai_result(self, plate_number: str):
        plate_number = re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', plate_number or "").upper()
        for key in self.plate_white_list.keys():
            now = time.time()
            pre_time = self.plate_white_list[key].get("last_matched", 0)
            _key = re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', key or "").upper()
            if Levenshtein.distance(plate_number, _key) <= 2 and now - pre_time > 10:
                print(f"Plate {plate_number} matched whitelist entry {key} with distance {Levenshtein.distance(plate_number, _key)}")
                self.plate_white_list[key]["last_matched"] = now
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.post(
                            "http://localhost:8087/barrier/open",
                            json={"io_pin": 5},
                            timeout=5.0,
                        )
                        # response.raise_for_status()
                        print(f"Barrier opened for plate {key}")
                    except httpx.HTTPError as e:
                        print(f"Failed to open barrier for plate {key}: {e}")




plate_white_list_service = PlateWhiteListService()
