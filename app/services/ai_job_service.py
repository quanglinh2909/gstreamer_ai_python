import json
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.models.ai_config import AIConfig
from app.repositories.ai_config_repository import AIRepository


def job_transform(ai_job: dict) -> Optional[str]:
    """Transform của tầng con đầu tiên trong một job engine trả về.

    Chỉ dùng để ĐOÁN loại AI của job mồ côi (không có dòng ai_configs) — cách
    nhận diện chính là job_id. Trước đây engine trả thẳng `transformData` ở
    mức job; giờ transform nằm trong từng tầng nên phải lần vào mảng stages."""
    for stage in (ai_job.get("stages") or [])[1:]:
        transform = (stage or {}).get("transform")
        if transform:
            return transform
    return None


@dataclass(frozen=True)
class AIStage:
    """MỘT tầng của cây model. Tầng đầu chạy trên cả khung hình; mỗi tầng sau
    chạy trên ảnh cắt ra từ từng detection của tầng cha.

    Cỡ đầu vào KHÔNG khai ở đây — engine đọc thẳng từ file .rknn."""

    # Tên file trong danh sách /ai-models của engine (không phải đường dẫn):
    # upsert tra ra path thật, vì path phụ thuộc thư mục weights của board.
    model_file: str
    model_type: str
    # Chỉ số tầng cha trong danh sách. None = nối vào tầng ngay trước (chuỗi
    # thẳng), tầng đầu tiên thì là chạy trên khung hình.
    parent: Optional[int] = None
    # Cách dựng ảnh đầu vào từ hộp của tầng cha ("" / None = cắt thẳng theo hộp).
    transform: Optional[str] = None
    # Lọc ĐẦU VÀO: chỉ nhận detection của tầng cha mang lớp này. Đây là chỗ để
    # "model 2 chỉ lấy biển số, model 3 chỉ lấy ô tô/xe máy/xe tải".
    input_classes: Optional[str] = None
    # Lọc ĐẦU RA: giữ lớp nào trong kết quả của CHÍNH tầng này. CSV id lớp kiểu
    # "0,1"; None/"" = giữ tất cả. Khớp parseClassFilter bên C++ (ai/Config.hpp).
    class_filter: Optional[str] = None
    # Ngưỡng điểm của tầng này. 0.2 là ngưỡng engine vẫn dùng xưa nay — lọc
    # chặt hơn là việc của phía Python (ai_configs.primary_conf), engine để
    # rộng để tracker còn nhìn thấy vật mờ.
    conf: float = 0.2


@dataclass(frozen=True)
class AIJobSpec:
    config_type: str
    name: str
    # Theo thứ tự chạy. Một model = một phần tử; thêm tầng là thêm phần tử.
    stages: tuple = ()

    @property
    def transform_data(self) -> Optional[str]:
        """Transform của tầng con đầu tiên có khai.

        Chỉ còn dùng để NHẬN RA job kiểu cũ (xem _find_existing) — hồi engine
        còn cứng hai model thì transform là thứ duy nhất phân biệt được job
        khuôn mặt với job biển số trên cùng một camera."""
        for stage in self.stages[1:]:
            if stage.transform:
                return stage.transform
        return None


@dataclass(frozen=True)
class AIVariant:
    """MỘT CÁCH LÀM của một loại AI.

    Cùng "nhận dạng biển số" nhưng có thể làm bằng nhiều bộ model khác nhau,
    và mỗi bộ lại tracking theo kiểu khác nhau. Biến thể gói TRỌN một cách
    làm: cây model + tracking theo lớp nào + lớp nào gắn vào lớp được track.

    Loại AI chỉ có một biến thể thì giao diện không hiện ô chọn (không có gì
    để chọn); từ hai trở lên mới hiện.

    TRACK/ATTACH giải bài toán chung: "vật CHÍNH là thứ cần bám theo thời
    gian, vật PHỤ chỉ là thuộc tính của nó trong khung hình này".
      * khẩu trang — track người (0), gắn có/không khẩu trang (3, 5) vào người
      * biển số    — track XE, gắn biển (5) vào xe; biển đi theo xe, không tự
                     sinh ra một track riêng nhảy loạn khi xe che khuất biển
    """

    id: str
    label: str
    spec: AIJobSpec
    # Lớp đưa vào tracker. None = track mọi lớp (vật chính chính là đầu ra
    # duy nhất của model, vd biển số tự nó là một track).
    track_classes: Optional[frozenset] = None
    # Lớp KHÔNG track, đem gắn vào box được track chứa nó nhiều nhất.
    attach_classes: frozenset = frozenset()
    # Phần diện tích của box phụ phải nằm trong box chính thì mới coi là của nó.
    attach_containment: float = 0.5
    # Tên/màu cho lớp trên overlay gỡ lỗi (thuần trang trí).
    class_meta: Optional[dict] = None

    @property
    def transform_data(self) -> Optional[str]:
        """Transform của tầng thứ hai — chỉ còn dùng để nhận diện job kiểu cũ
        (xem _find_existing) và gán nhãn loại AI cho job mồ côi."""
        for stage in self.stages[1:]:
            if stage.transform:
                return stage.transform
        return None


class AIJobService:
    @staticmethod
    def _get_path(ai_models, file_name):
        return next(
            (m["path"] for m in ai_models if m["fileName"] == file_name),
            None,
        )

    @staticmethod
    def _find_existing(ai_jobs, spec):
        """Locate the C++-side job that already represents this SPEC for
        this camera so we PUT (update) instead of POSTing a duplicate.

        Match by `name` first — it's unique per SPEC (face/plate/
        restricted_area each carry a distinct label) and works even for
        single-stage jobs whose `transformData` is empty. The C++ DTO
        serialises a missing transform as "" not null, so a naive
        `transformData == None` comparison always failed for
        restricted_area and silently created a new duplicate every save.

        Fall back to a transform match for backwards-compat with stage-2
        cascade jobs (face / plate) that were saved before the rename."""
        by_name = next(
            (j for j in ai_jobs if j.get("name") == spec.name), None,
        )
        if by_name is not None:
            return by_name
        if spec.transform_data:
            return next(
                (j for j in ai_jobs
                 if job_transform(j) == spec.transform_data),
                None,
            )
        return None

    @staticmethod
    def stage_preview(stage: AIStage, index: int) -> dict:
        """AIStage -> tầng ở dạng engine hiểu, NHƯNG giữ `modelFile` thay cho
        `modelPath`.

        Dùng cho /ai-variants: trang thử model cần thấy đúng cây model mà
        camera đang chạy để nạp vào form, mà đường dẫn thật thì phụ thuộc thư
        mục weights của board — nó tự tra qua /ai-models của engine. Tách ra
        khỏi `_stage_payload` để phần suy ra `parent` chỉ viết một lần: sai lệch
        giữa "cây trang thử vẽ" và "cây camera chạy" là loại lỗi rất khó thấy."""
        out = {
            "modelFile": stage.model_file,
            "modelType": stage.model_type,
            "conf": stage.conf,
            # Tầng 0 luôn chạy trên khung hình; mặc định còn lại là chuỗi thẳng.
            "parent": stage.parent if stage.parent is not None
            else (-1 if index == 0 else index - 1),
        }
        if stage.transform:
            out["transform"] = stage.transform
        if stage.input_classes is not None:
            out["inputClasses"] = stage.input_classes
        if stage.class_filter is not None:
            out["classFilter"] = stage.class_filter
        return out

    @staticmethod
    def _stage_payload(ai_models, stage: AIStage, index: int) -> dict:
        """AIStage -> một phần tử của mảng `stages` mà engine nhận."""
        out = AIJobService.stage_preview(stage, index)
        out["modelPath"] = AIJobService._get_path(ai_models, out.pop("modelFile"))
        return out

    @classmethod
    def _stages_payload(cls, ai_models, spec: AIJobSpec) -> list:
        return [cls._stage_payload(ai_models, s, i)
                for i, s in enumerate(spec.stages)]

    @staticmethod
    def _to_ratio(value: float) -> float:
        return value / 100.0 if value > 1 else value

    async def upsert(self, db: AsyncSession, req, spec: AIJobSpec, extra_data=None):
        req.primaryConf = self._to_ratio(req.primaryConf)
        req.secondaryConf = self._to_ratio(req.secondaryConf)

        ai_jobs = await HTTPXClient.get(f"/cameras/{req.cameraId}/ai-jobs")
        existing = self._find_existing(ai_jobs, spec)
        ai_models = await HTTPXClient.get("/ai-models")

        # Dựng THẲNG payload của engine thay vì model_dump() rồi ghi đè: DTO
        # phía Python mang một đống trường engine không biết (polygons, tracker,
        # dwellSeconds, min_plate_length...) và chúng nằm lại trong ai_configs.
        # Engine chỉ cần đúng 5 trường này.
        payload = {
            "name": spec.name,
            "cameraId": req.cameraId,
            "enabled": getattr(req, "enabled", True),
            "maxFps": getattr(req, "maxFps", 0),
            "stages": self._stages_payload(ai_models, spec),
        }

        if existing:
            data = await HTTPXClient.put(f"/ai-jobs/{existing['id']}", json=payload)
        else:
            data = await HTTPXClient.post("/ai-jobs", json=payload)

        await AIRepository.create_or_update(
            db,
            AIConfig(
                camera_id=req.cameraId,
                type=spec.config_type,
                polygons=req.polygons,
                job_id=data.get("id"),
                primary_conf=req.primaryConf,
                secondary_conf=req.secondaryConf,
                fps=req.maxFps,
                tracker=req.tracker,
                overlap_threshold=req.overlap_threshold,
                dwell_seconds=req.dwellSeconds,
                extra_data=extra_data,
                # getattr: DTO cũ chưa khai báo trường này vẫn chạy được.
                save_detections=bool(getattr(req, "saveDetections", False)),
                # Mặc định BẬT, và client cũ không gửi trường này cũng phải ra
                # BẬT — nếu không, một bản giao diện cũ lưu lại cấu hình là âm
                # thầm tắt việc ghi sự kiện của camera đó.
                save_events=bool(getattr(req, "saveEvents", True)),
            ),
        )
        # Force the recv loop to reload tracker/polygons/thresholds for this
        # (camera, job) instead of keeping the stale snapshot it cached on
        # the first inbound message. Lazy import — avoids the
        # ai_job_service ↔ process_ai_hepper ↔ face_recognition_service cycle.
        from app.services.process_ai_service import process_ai_service
        process_ai_service.invalidate(req.cameraId, data.get("id"))
        return data

    async def inference_model(self, image: tuple, stages: list):
        """Chạy một cây model trên MỘT tấm ảnh.

        `stages` đúng dạng mà engine nhận (mảng dict modelPath/modelType/...),
        nên trang thử model gửi được cây bao nhiêu tầng tuỳ ý mà không phải sửa
        gì ở đây."""
        files = {"image": image}
        data = {"stages": json.dumps(stages)}
        return await HTTPXClient.post("/inference/run", data=data, files=files)

    async def inference_with_spec(
        self,
        image: tuple,
        spec: AIJobSpec,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        """Chạy đúng cây model mà spec này sẽ chạy khi bật trên camera.

        primary_conf/secondary_conf ghi đè ngưỡng của tầng 0 và các tầng sau —
        thử ảnh thường muốn nới ngưỡng hơn lúc chạy thật."""
        ai_models = await HTTPXClient.get("/ai-models")
        stages = self._stages_payload(ai_models, spec)
        for i, stage in enumerate(stages):
            stage["conf"] = primary_conf if i == 0 else secondary_conf
        return await self.inference_model(image=image, stages=stages)


ai_job_service = AIJobService()
