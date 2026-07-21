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

    # Detector confidence used when registering a face. Registration goes
    # through the C++ engine's /inference/run with FACE_SPEC — the same
    # models the live RTSP pipeline uses — so the model paths live on the
    # engine side (GET /ai-models), not here.
    FACE_DETECT_CONF: Optional[float] = 0.5
    IS_OPEN_DOOR_WHEN_FACE_MASK: Optional[bool] = True

    class Config:
        env_file = ".env"


settings = Settings()
