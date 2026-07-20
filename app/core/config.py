from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    AI_API_BASE_URL: Optional[str] = "http://localhost:8009"
    # Bluetooth "detect" service: notified with the matched identity's MAC so it
    # can correlate the BLE beacon with the parking-lot event.
    BLE_DETECT_URL: Optional[str] = None
    PORT: Optional[int] = 8010
    MILVUS_URI: Optional[str] = "./milvus_face.db"
    MILVUS_FACE_COLLECTION: Optional[str] = "face_embeddings"
    FACE_EMBEDDING_DIM: Optional[int] = 512

    # Face registration runs entirely in Python (YOLO pose → align → AdaFace
    # RKNN) so the registration embedding matches the proven ai_result_face.py
    # pipeline, independent of the C++ inference path used by live RTSP.
    FACE_DETECTOR_MODEL_PATH: Optional[str] = \
        "/home/orangepi/face_inspireface/weight/yolov8n-face_rknn_model_640"
    FACE_EMBEDDING_MODEL_PATH: Optional[str] = \
        "/home/orangepi/Documents/test/weights/adaface_ir101_fp16.rknn"
    FACE_DETECT_CONF: Optional[float] = 0.5
    IS_OPEN_DOOR_WHEN_FACE_MASK: Optional[bool] = True

    class Config:
        env_file = ".env"


settings = Settings()
