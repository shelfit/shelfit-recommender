from enum import StrEnum
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str
    embedding_model: str

class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    openrouter_base_uri: str
    openrouter_api_key: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    qdrant_settings: QdrantSettings = Field(default_factory=QdrantSettings)
    openrouter_settings: OpenRouterSettings = Field(default_factory=OpenRouterSettings)

class RecommendRequest(BaseModel):
    query: str

class IntentItemType(StrEnum):
    AUTHOR = 'author'
    BOOK = 'book'

class IntentType(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    SIMILAR = "similar"

class IntentTerm(BaseModel):
    model_config = ConfigDict(extra='forbid')

    item: str
    item_type: IntentItemType
    intent: IntentType

class ParsedQueryIntent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    query_full: str
    query_context_residue: str
    terms: List[IntentTerm]