from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str
    embedding_model: str

class RecommendRequest(BaseModel):
    query: str