from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str = "books"
    embedding_model: str = "all-MiniLM-L6-v2"

    openrouter_base_uri: str
    openrouter_api_key: str

    gateway_mysql_url: str


settings = Settings() # noqa