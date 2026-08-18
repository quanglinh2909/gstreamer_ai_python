import asyncio
import os
import sys
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from app.core.config import settings
from app.dto.face_recognition_dto import FaceRecognitionDTO
from app.dto.identity_dto import FaceInfo
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_face import EventFace
from app.models.identity import Identity
from app.repositories.event_face_repository import EventFaceRepository
from app.repositories.face_vector_repository import FaceVectorRepository
from app.services.ai_job_service import AIJobSpec, AIStage, AIVariant, ai_job_service
# UPLOADS_ROOT is re-exported here: identity_service imports it from this
# module, and it now lives with the shared crop/save helpers.
from app.services.ai_service_base import UPLOADS_ROOT, AIServiceBase
from app.services.parking_lot_service import parking_lot_service
from app.tasks.task_parking_lot import task_parking_lot
from app.ws.face_event_ws import face_event_broadcaster

# Per-tracker identification state. entered_zone seeds it, in_the_area drives
# the retry loop, exited_zone clears it. Keyed by (camera_id, tracker_id).
_MATCHING = "matching"   # a match task is in flight; skip new attempts
_PENDING = "pending"     # last attempt failed; try again on the next frame
_RESOLVED = "resolved"   # successfully identified; no more attempts

FACE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.FACE_RECOGNITION.value,
    name="Face recognition",
    stages=(
        # Tầng 0: tìm mặt + 5 điểm mốc trên cả khung.
        AIStage(model_file="yolov8_pose_face_in8.rknn", model_type="yolov8_pose"),
        # Tầng 1: nắn mặt theo điểm mốc rồi trích vector nhận dạng.
        AIStage(model_file="adaface_ir101_fp16.rknn",
                model_type="face_recognition", transform="align_face"),
    ),
)

# Một cách làm duy nhất: mặt tự nó là vật được bám, không có lớp phụ nào gắn
# vào. Vẫn khai thành biến thể để tracking/overlay đọc từ cùng một chỗ với ba
# loại AI kia; giao diện sẽ không hiện ô chọn vì chỉ có một.
FACE_VARIANT = AIVariant(
    id="yolov8_pose_adaface",
    label="YOLOv8-pose + AdaFace",
    spec=FACE_SPEC,
)


class FaceRecognitionService(AIServiceBase):
    VARIANTS = (FACE_VARIANT,)

    EVENT_FOLDER = "faces"

    # MỘT SỰ KIỆN CHO MỘT TRACK — mất track là coi như người đó đi.
    #
    # Điểm chốt phải là ĐỜI CỦA TRACKER ID, KHÔNG phải nhịp entered/exited
    # zone. Hai thứ đó không đi cùng nhau: đo trên camera "test", tracker giữ
    # nguyên `id=0` suốt mấy phút liền trong khi vùng bắn ra/vào 28/22 lần và
    # đẻ ra 65 sự kiện. Lý do là khung hình nào model không bắt được mặt thì id
    # biến mất khỏi danh sách "đang trong vùng"; đủ `exit_grace` khung (chỉ 2
    # giây với bytetrack ở fps 5) là `exited_zone` bắn, xoá trạng thái, rồi
    # khung sau mặt hiện lại -> `entered_zone` -> thêm một sự kiện nữa, dù
    # tracker chưa hề đánh mất người đó. Chính chú thích trong
    # `ProcessAiHepper.lost_buffer_frames` đã cảnh báo đúng ca này.
    #
    # Nên: nhớ theo `(cameraId, tracker_id)` và KHÔNG xoá khi rời vùng. Chỉ
    # quên khi id đó im lặng quá _TRACK_TTL_S — lúc ấy tracker chắc chắn đã
    # bỏ nó (bộ đệm của tracker chỉ ~2-3 giây), id sau là người mới thật.
    _TRACK_TTL_S = 10

    def __init__(self):
        # (cameraId, tracker_id) -> one of _MATCHING / _PENDING / _RESOLVED.
        # Survives across frames of the same tracker; cleared on exited_zone.
        self._track_state: dict = {}
        # (cameraId, tracker_id) -> lần cuối thấy id này (còn sống hay không).
        self._track_seen: dict = {}
        # Những track đã ghi sự kiện rồi.
        self._written_tracks: set = set()
        self._last_prune = 0.0

    def _touch_track(self, camera_id, tid, now: float) -> None:
        """Đánh dấu tracker id còn sống, và quên hẳn những id đã chết."""
        key = (str(camera_id), int(tid))
        last = self._track_seen.get(key)
        # Xét CHÍNH id này trước, đừng chờ đợt dọn định kỳ: id im lặng quá lâu
        # nghĩa là tracker đã bỏ nó và vừa cấp lại cho lần xuất hiện khác, nên
        # phải quên sạch rồi mới đóng dấu. Làm ngược lại là tự cứu sống nó.
        if last is not None and now - last > self._TRACK_TTL_S:
            self._forget_track(key)
        self._track_seen[key] = now

        # Đợt dọn định kỳ chỉ để bảng khỏi phình với những id không bao giờ
        # quay lại.
        if now - self._last_prune < self._TRACK_TTL_S:
            return
        self._last_prune = now
        for k in [k for k, t in self._track_seen.items()
                  if now - t > self._TRACK_TTL_S]:
            self._track_seen.pop(k, None)
            self._forget_track(k)

    def _forget_track(self, key) -> None:
        self._written_tracks.discard(key)
        self._track_state.pop(key, None)

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

    @staticmethod
    def _pick_largest_face(result):
        """Biggest detection that actually produced an embedding, as
        (embedding, bbox, score). None when the image has no usable face.

        Mirrors the old Python pipeline's "largest face wins" rule; a
        detection without an embedding means align/AdaFace failed for it,
        so it can't be registered."""
        best = None
        best_area = 0.0
        for d in (result or {}).get("detections") or []:
            embedding = FaceRecognitionService._extract_embedding(d)
            if not embedding:
                continue
            x1, y1 = float(d.get("x1", 0.0)), float(d.get("y1", 0.0))
            x2, y2 = float(d.get("x2", 0.0)), float(d.get("y2", 0.0))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (embedding, (x1, y1, x2, y2), float(d.get("score", 0.0)))
        return best

    async def register_face(self, identity_id: int, image: tuple) -> FaceInfo:
        """Register a face through the C++ engine's one-shot inference
        endpoint — the same YOLOv8-pose → align_face → AdaFace path the live
        camera pipeline runs.

        This used to run a separate Python pipeline (ultralytics YOLO +
        RKNN AdaFace), which dragged torch into the backend for this single
        endpoint and — more importantly — detected faces with a *different*
        model than the live path, so a registered embedding came from a
        slightly different crop than the one it would later be matched
        against. Reusing the engine removes both problems; `test_inference`
        above already goes through the same call."""
        image_bytes = image[1] if len(image) >= 2 else None
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image")

        result = await ai_job_service.inference_with_spec(
            image=image,
            spec=FACE_SPEC,
            primary_conf=settings.FACE_DETECT_CONF or 0.5,
            # Face embedding produces no stage-2 children, so there is
            # nothing for a secondary threshold to filter.
            secondary_conf=0.0,
        )

        face = self._pick_largest_face(result)
        if face is None:
            raise HTTPException(
                status_code=400, detail="No face/embedding detected in image"
            )
        embedding, bbox, score = face

        # FaceVectorRepository L2-normalises on both insert and search, so the
        # engine's raw AdaFace output (already ~unit norm) needs no extra work.
        milvus_id = await asyncio.to_thread(
            FaceVectorRepository.insert,
            embedding=embedding,
            identity_id=identity_id,
        )

        det_for_crop = {
            "x1": bbox[0], "y1": bbox[1],
            "x2": bbox[2], "y2": bbox[3],
        }
        paths = await asyncio.to_thread(
            self._save_identity_images_blocking,
            image_bytes,
            identity_id,
            det_for_crop,
        )
        full_url, crop_url, _box = (paths if paths is not None else (None, None, None))

        return FaceInfo(
            id=milvus_id,
            score=score,
            embedding=embedding,
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
        crop = cls._make_crop(
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
    # When the padded box is wider than target, the extra room goes down
    # into the body, not up above the head.
    CROP_VERTICAL_BIAS = "below"

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

    EVENT_MODEL = EventFace
    EVENT_BROADCASTER = face_event_broadcaster
    EVENT_SOURCE = "face_recognition"

    async def _persist_event(
        self, meta, parent, full_jpeg, tid, timestamp, secondary_conf,
        save_unmatched: bool,
        persist: bool = True,
        extra_data=None,
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
            # Ngưỡng "độ chính xác khuôn mặt" của CHÍNH cổng này (bãi xe), chọn
            # từ giao diện; camera không thuộc bãi nào thì giữ mặc định 0.15.
            lot = parking_lot_service.get_by_camera_id(meta["cameraId"])
            face_conf = (
                float(lot["face_confidence"])
                if lot and lot.get("face_confidence") is not None
                else 0.15
            )
            _identity_id, _similarity = await self._match_identity(embedding, face_conf)
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

        # Track này đã ghi sự kiện rồi: vẫn trả kết quả khớp để nó ở nguyên
        # trạng thái RESOLVED, nhưng không ghi thêm dòng nào nữa.
        if not persist:
            return identity_id

        if not identity_id:
            return None

        # Track này đã có sự kiện rồi. Đây mới là chốt chặn thật: nó SỐNG SÓT
        # qua nhịp exited/entered zone, thứ mà `_track_state` thì không.
        write_key = (str(meta["cameraId"]), int(tid))
        if write_key in self._written_tracks:
            return identity_id

        # Tên người tra TRƯỚC khi ghi, để gói WebSocket mang đủ id + nhãn mà
        # không phải mở lại session sau commit.
        identity_name = None
        async with session_factory() as db:
            identity_name = await db.scalar(
                select(Identity.name).where(Identity.id == identity_id)
            )

        # Ghi ảnh + hàng + WebSocket + đánh thức ghi hình: AIServiceBase.
        # confidence ở đây là ĐỘ GIỐNG với người đã đăng ký, không phải điểm
        # phát hiện của box.
        event = await self.save_event(
            meta, parent, full_jpeg, tid, timestamp,
            columns={"identity_id": identity_id},
            payload={"identity_id": identity_id, "name": identity_name},
            confidence=similarity,
            extra_data=extra_data,
        )
        if event is None and self.should_save_events(extra_data):
            # Ghi ảnh hỏng -> trả None để track ở lại PENDING và khung sau thử
            # lại. Trả identity_id ở đây là track thành RESOLVED và lần xuất
            # hiện này mất luôn cả sự kiện lẫn ảnh.
            #
            # Camera TẮT ghi sự kiện thì save_event cũng trả None (đúng: không
            # có hàng nào) nhưng đó là kết quả MONG MUỐN — không tách hai
            # trường hợp thì track đứng mãi ở PENDING và khớp lại mỗi khung.
            return None
        self._written_tracks.add(write_key)
        return identity_id

    async def _run_match(
        self, meta, parent, full_jpeg, tid, timestamp, secondary_conf,
        save_unmatched: bool,
        persist: bool = True,
        extra_data=None,
    ):
        key = (str(meta["cameraId"]), int(tid))
        try:
            identity_id = await self._persist_event(
                meta, parent, full_jpeg, tid, timestamp, secondary_conf,
                save_unmatched=save_unmatched,
                persist=persist,
                extra_data=extra_data,
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

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        self._touch_track(meta["cameraId"], id, timestamp)
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
                            secondary_conf, save_unmatched=True,
                            extra_data=extra_data)
        )

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        self._touch_track(meta["cameraId"], id, timestamp)
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
                            persist=not already_resolved,
                            extra_data=extra_data)
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        # CỐ Ý không xoá gì ở đây. Vùng bắn "đã ra" chỉ vì vài khung hình
        # không bắt được mặt, trong khi tracker vẫn giữ nguyên id — xoá ở đây
        # là khung sau vào lại và đẻ thêm một sự kiện trùng. Việc quên một
        # track do `_touch_track` lo, dựa trên id đó im lặng bao lâu.
        print(f"Face exited_zone id={id}")


face_recognition_service = FaceRecognitionService()
