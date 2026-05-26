from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    AI_API_BASE_URL: Optional[str] = "http://localhost:8009"
    PORT: Optional[int] = 8010

    class Config:
        env_file = ".env"


settings = Settings()
