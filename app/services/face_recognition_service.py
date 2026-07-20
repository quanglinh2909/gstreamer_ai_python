import asyncio
import datetime
import os
import sys
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from app.dto.face_recognition_dto import FaceRecognitionDTO
from app.dto.identity_dto import FaceInfo
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_face import EventFace
from app.models.identity import Identity
from app.repositories.event_face_repository import EventFaceRepository
from app.repositories.face_vector_repository import FaceVectorRepository
from app.services.ai_job_service import AIJobSpec, ai_job_service
from app.tasks.task_parking_lot import task_parking_lot
from app.utils import face_embedder
from app.utils.image_crop import fixed_size_crop
from app.ws.face_event_ws import face_event_broadcaster

# Per-tracker identification state. entered_zone seeds it, in_the_area drives
# the retry loop, exited_zone clears it. Keyed by (camera_id, tracker_id).
_MATCHING = "matching"   # a match task is in flight; skip new attempts
_PENDING = "pending"     # last attempt failed; try again on the next frame
_RESOLVED = "resolved"   # successfully identified; no more attempts

FACE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.FACE_RECOGNITION.value,
    transform_data="align_face",
    name="Face recognition",
    model_file_1="yolov8_pose_face_in8.rknn",
    model_file_2="adaface_ir101_fp16.rknn",
    model_type_1="yolov8_pose",
    model_type_2="face_recognition",
)

UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class FaceRecognitionService:
    # (cameraId, tracker_id) -> timestamp of the last EventFace written for
    # that tracker. _track_state already prevents duplicate rows while a
    # tracker stays continuously in the zone; this guards the exit/re-enter
    # case (brief occlusion) where _track_state is cleared but the tracker
    # keeps its id, which would otherwise write a second row.
    _REENTER_COOLDOWN_S = 15

    def __init__(self):
        # (cameraId, tracker_id) -> one of _MATCHING / _PENDING / _RESOLVED.
        # Survives across frames of the same tracker; cleared on exited_zone.
        self._track_state: dict = {}
        self._last_saved: dict = {}

    async def face_recognition(self, db: AsyncSession, req: FaceRecognitionDTO):
        return await ai_job_service.upsert(db, req, FACE_SPEC)

    async def test_inference(
        self,
        image: tuple,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        """Diagnostic: run face detection + embedding on the uploaded image,
        then show top matches against every registered identity. Use this to
        verify whether a registered person's own photo self-matches at high
        score (should be > 0.7) and whether strangers stay low (< 0.4)."""
        result = await ai_job_service.inference_with_spec(
            image=image,
            spec=FACE_SPEC,
            primary_conf=primary_conf,
            secondary_conf=secondary_conf,
        )
        detections = (result or {}).get("detections") or []
        for d in detections:
            emb = self._extract_embedding(d)
            if not emb:
                d["matches"] = []
                continue
            hits = await asyncio.to_thread(
                FaceVectorRepository.search, embedding=emb, top_k=5,
            )
            d["matches"] = [
                {"identity_id": h.get("identity_id"), "score": round(float(h.get("score", 0.0)), 4)}
                for h in hits
            ]
            # Drop the embedding from the response — too noisy for visual debug.
            d.pop("embedding", None)
        return result

    async def list_events(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        return await EventFaceRepository.list_paginated(db, page, size, camera_id)

    async def register_face(self, identity_id: int, image: tuple) -> FaceInfo:
        """Register a face by extracting an embedding via the standalone Python
        pipeline (YOLO pose → AdaFace align → AdaFace RKNN), matching the
        reference ai_result_face.py. Independent from the C++ live inference
        path so registration quality is not affected by camera/decode quirks."""
        image_bytes = image[1] if len(image) >= 2 else None
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image")

        face = await asyncio.to_thread(face_embedder.extract_face_embedding, image_bytes)
        if face is None:
            raise HTTPException(
                status_code=400, detail="No face/embedding detected in image"
            )

        milvus_id = await asyncio.to_thread(
            FaceVectorRepository.insert,
            embedding=face.embedding,
            identity_id=identity_id,
        )

        det_for_crop = {
            "x1": face.bbox[0], "y1": face.bbox[1],
            "x2": face.bbox[2], "y2": face.bbox[3],
        }
        paths = await asyncio.to_thread(
            self._save_identity_images_blocking,
            image_bytes,
            identity_id,
            det_for_crop,
        )
        full_url, crop_url = (paths if paths is not None else (None, None))

        return FaceInfo(
            id=milvus_id,
            score=face.score,
            embedding=face.embedding,
            image_full=full_url,
            image_crop=crop_url,
        )

    @classmethod
    def _save_identity_images_blocking(cls, image_bytes, identity_id, parent):
        folder_rel = os.path.join("identities", str(identity_id))
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        full_abs = os.path.join(folder_abs, "full.jpg")
        crop_abs = os.path.join(folder_abs, "crop.jpg")

        with open(full_abs, "wb") as fp:
            fp.write(image_bytes)

        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return f"/uploads/{folder_rel}/full.jpg", None
        crop = cls._make_face_crop(
            img,
            float(parent.get("x1", 0.0)), float(parent.get("y1", 0.0)),
            float(parent.get("x2", 0.0)), float(parent.get("y2", 0.0)),
        )
        if crop is None or not cv2.imwrite(crop_abs, crop):
            return f"/uploads/{folder_rel}/full.jpg", None
        return (
            f"/uploads/{folder_rel}/full.jpg",
            f"/uploads/{folder_rel}/crop.jpg",
        )

    @staticmethod
    def _find_parent(meta, tid):
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    # Face bbox is tight around the face. Pad outward when saving the event
    # crop so the picture includes hair, neck and shoulders — easier to read
    # by a human reviewer than a tight head-only crop. Final crop covers
    # ~2.4× face width and ~3.4× face height (full head + upper torso).
    CROP_PAD_LEFT = 0.7
    CROP_PAD_RIGHT = 0.7
    CROP_PAD_TOP = 0.7
    CROP_PAD_BOTTOM = 1.7

    # Every saved face crop is rendered at exactly this size. Portrait 4:5
    # roughly matches the natural face+shoulders aspect (1.8 wide × 2.4
    # tall after padding), so the crop is grown — not stretched — to fit.
    CROP_OUTPUT_W = 224
    CROP_OUTPUT_H = 280
    CROP_PAD_COLOR = 114  # neutral grey for letterbox fill (matches YOLO)

    @classmethod
    def _make_face_crop(cls, img, bx1, by1, bx2, by2):
        # vertical_bias="below" — when the padded box is wider than
        # target, the extra room goes down into the body, not up above
        # the head. Matches the old face-specific behaviour.
        return fixed_size_crop(
            img, bbox=(bx1, by1, bx2, by2),
            pad_lrtb=(cls.CROP_PAD_LEFT, cls.CROP_PAD_RIGHT,
                      cls.CROP_PAD_TOP, cls.CROP_PAD_BOTTOM),
            output_size=(cls.CROP_OUTPUT_W, cls.CROP_OUTPUT_H),
            pad_color=cls.CROP_PAD_COLOR,
            vertical_bias="below",
        )

    @classmethod
    def _save_images_blocking(cls, full_jpeg, meta, parent, tid):
        if not full_jpeg:
            return None
        img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        crop = cls._make_face_crop(
            img,
            float(parent["x1"]), float(parent["y1"]),
            float(parent["x2"]), float(parent["y2"]),
        )
        if crop is None:
            return None

        date = datetime.date.today().isoformat()
        folder_rel = os.path.join("faces", str(meta["cameraId"]), date)
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        stem = f"{int(meta['seq']):010d}_{int(tid)}"
        full_abs = os.path.join(folder_abs, f"{stem}_full.jpg")
        crop_abs = os.path.join(folder_abs, f"{stem}_crop.jpg")

        with open(full_abs, "wb") as fp:
            fp.write(full_jpeg)
        if not cv2.imwrite(crop_abs, crop):
            return None
        return (
            f"/uploads/{folder_rel}/{stem}_full.jpg",
            f"/uploads/{folder_rel}/{stem}_crop.jpg",
        )

    @staticmethod
    def _extract_embedding(parent):
        for key in ("embedding", "feature", "features"):
            vec = parent.get(key)
            if vec:
                return list(vec)
        return None

    @staticmethod
    async def _match_identity(embedding, secondary_conf):
        """Returns (identity_id, similarity). identity_id is None when no hit
        passes secondary_conf; similarity is the top score regardless, so the
        caller can persist it even on a miss."""
        # In-memory cosine search (cache mirror of Milvus) — no gRPC round-trip
        # on the per-frame hot path, so continuous matching stays cheap.
        hits = FaceVectorRepository.search_cached(embedding=embedding, top_k=3)
        if not hits:
            return None, 0.0
        # Print top hits so we can see best vs runner-up gap. A small gap means
        # the embedding is close to multiple identities — likely mis-ID risk.
        # print("[face match] " + " | ".join(
        #     f"id={h.get('identity_id')} score={h.get('score', 0):.3f}" for h in hits
        # ))
        top = hits[0]
        similarity = float(top.get("score", 0.0))
        if similarity < secondary_conf:
            return None, similarity
        identity_id = top.get("identity_id")
        return (int(identity_id) if identity_id is not None else None), similarity

    async def _persist_event(
        self, meta, parent, full_jpeg, tid, timestamp, secondary_conf,
        save_unmatched: bool,
        persist: bool = True,
    ):
        """Run identification and (optionally) write one EventFace row.

        Returns the matched identity_id, or None if no match passed the
        threshold. When `save_unmatched` is False, an unmatched result is
        not written to the DB — the caller keeps the tracker in PENDING
        state so the next frame retries. When `persist` is False the match
        still runs (so the caller can keep re-identifying every frame) but
        nothing is written to the DB — used once the tracker is already
        RESOLVED to avoid duplicate EventFace rows."""
        from app.services.process_ai_service import process_ai_service
        session_factory = process_ai_service._session_factory
        if session_factory is None:
            return None

        identity_id = None
        similarity = 0.0
        embedding = self._extract_embedding(parent)
        if embedding:
            _identity_id, _similarity = await self._match_identity(embedding, 0.15)
            if _identity_id is not None:
                task_parking_lot.add_task({
                    "task": "face_recognition",
                    "identity_id": _identity_id,
                    # "similarity": _similarity,
                    "timestamp": timestamp,
                    "camera_id": meta["cameraId"],
                    "full_jpeg": full_jpeg,
                })
            if _similarity >= secondary_conf:
                identity_id, similarity = _identity_id, _similarity

        if identity_id is None and not save_unmatched:
            return None

        # Already identified on a previous frame: keep returning the match so
        # the tracker stays RESOLVED, but don't write another EventFace row.
        if not persist:
            return identity_id

        if not identity_id:
            return None

        # Exit/re-enter dedup: the same tracker may briefly leave the zone
        # (occlusion) which clears _track_state, then re-enter under the same
        # id. Skip writing another row if we saved one for this tracker very
        # recently — but still return the match so it stays RESOLVED.
        key = (str(meta["cameraId"]), int(tid))
        last = self._last_saved.get(key)
        if last is not None and timestamp - last < self._REENTER_COOLDOWN_S:
            return identity_id

        try:
            paths = await asyncio.to_thread(
                self._save_images_blocking, full_jpeg, meta, parent, tid,
            )
            if paths is None:
                return identity_id
            full_url, crop_url = paths

            async with session_factory() as db:
                event = EventFace(
                    camera_id=str(meta["cameraId"]),
                    identity_id=identity_id,
                    confidence=similarity,
                    timestamp=int(timestamp),
                    image_full=full_url,
                    image_crop=crop_url,
                )
                db.add(event)
                await db.commit()
                self._last_saved[key] = timestamp

                # Resolve the identity name in the same session so the
                # broadcast carries everything a UI needs (id + label),
                # then fan out to any WebSocket subscribers. expire_on_commit
                # is False on this session_factory, so `event.id` stays
                # populated after the commit without a refresh.
                identity_name = None
                if identity_id is not None:
                    identity_name = await db.scalar(
                        select(Identity.name).where(Identity.id == identity_id)
                    )
            face_event_broadcaster.publish({
                "id": event.id,
                "camera_id": event.camera_id,
                "identity_id": event.identity_id,
                "name": identity_name,
                "confidence": float(event.confidence),
                "timestamp": int(event.timestamp),
                "image_full": event.image_full,
                "image_crop": event.image_crop,
            })
        except Exception as exc:
            print(f"face persist error: {exc}", file=sys.stderr)
        return identity_id

    async def _run_match(
        self, meta, parent, full_jpeg, tid, timestamp, secondary_conf,
        save_unmatched: bool,
        persist: bool = True,
    ):
        key = (str(meta["cameraId"]), int(tid))
        try:
            identity_id = await self._persist_event(
                meta, parent, full_jpeg, tid, timestamp, secondary_conf,
                save_unmatched=save_unmatched,
                persist=persist,
            )
            # If the tracker has already been cleared by exited_zone while
            # we were matching, don't resurrect the entry.
            if key not in self._track_state:
                return
            self._track_state[key] = _RESOLVED if identity_id is not None else _PENDING
        except Exception:
            if key in self._track_state:
                self._track_state[key] = _PENDING
            raise

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id))
        # Mark _MATCHING synchronously so any in_the_area firing for the
        # same tracker on the next frame skips while we're still resolving.
        self._track_state[key] = _MATCHING
        print(f"entered_zone id={id}")
        asyncio.create_task(
            self._run_match(meta, parent, full_jpeg, id, timestamp,
                            secondary_conf, save_unmatched=True)
        )

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        key = (str(meta["cameraId"]), int(id))
        # Keep re-identifying every frame, but skip while a match is already in
        # flight to avoid spawning overlapping tasks. RESOLVED keeps running —
        # the match just won't be persisted again (persist=False below).
        if self._track_state.get(key) == _MATCHING:
            return
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        # Already identified once -> run the match but don't write a duplicate
        # EventFace row.
        already_resolved = self._track_state.get(key) == _RESOLVED
        self._track_state[key] = _MATCHING
        asyncio.create_task(
            self._run_match(meta, parent, full_jpeg, id, timestamp,
                            secondary_conf, save_unmatched=False,
                            persist=not already_resolved)
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        self._track_state.pop((str(meta["cameraId"]), int(id)), None)
        # Keep recent save stamps for the re-enter cooldown; drop aged-out
        # ones so the map can't grow without bound.
        cutoff = timestamp - self._REENTER_COOLDOWN_S
        self._last_saved = {k: t for k, t in self._last_saved.items() if t >= cutoff}
        print(f"Face exited_zone")


face_recognition_service = FaceRecognitionService()
