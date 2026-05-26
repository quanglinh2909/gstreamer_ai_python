from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    AI_API_BASE_URL: Optional[str] = "http://localhost:8009"
    PORT: Optional[int] = 8010
    MILVUS_URI: Optional[str] = "./milvus_face.db"
    MILVUS_FACE_COLLECTION: Optional[str] = "face_embeddings"
    FACE_EMBEDDING_DIM: Optional[int] = 512

    class Config:
        env_file = ".env"


settings = Settings()
