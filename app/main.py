from functools import lru_cache
from typing import Annotated
from fastapi import FastAPI, Depends
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.models import RecommendRequest, Settings, ParsedQueryIntent
from app.services.intent_parser_service import IntentParserService
from app.services.recommendation_service import RecommendationService

app = FastAPI()

@lru_cache
def get_settings():
    return Settings()

def get_openai_client(settings: Annotated[Settings, Depends(get_settings)]):
    return OpenAI(
        # base_url=settings.openrouter_settings.openrouter_base_uri,
        api_key=settings.openrouter_settings.openrouter_api_key,
    )

def get_intent_parser_service(client: Annotated[OpenAI, Depends(get_openai_client)]):
    return IntentParserService(client)

def get_recommendation_service(intent_parser_service: Annotated[IntentParserService, Depends(get_intent_parser_service)]):
    return RecommendationService(intent_parser_service)

def get_embedding_model(settings: Annotated[Settings, Depends(get_settings)]):
    return SentenceTransformer(settings.qdrant_settings.embedding_model)

def get_qdrant_client(settings: Annotated[Settings, Depends(get_settings)]):
    return QdrantClient(host=settings.qdrant_settings.qdrant_host, port=settings.qdrant_settings.qdrant_port)

@app.post("/api/recommend")
def recommend(
    request: RecommendRequest,
    model: Annotated[SentenceTransformer, Depends(get_embedding_model)],
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    embedding = model.encode(request.query)

    search_result = client.query_points(
        collection_name=settings.qdrant_settings.qdrant_collection_name,
        query=embedding.tolist(),
        limit=10
    )

    return search_result

@app.post("/api/intent-parse")
def intent_parse(
    request: RecommendRequest,
    recommendation_service: Annotated[RecommendationService, Depends(get_recommendation_service)],
):
    return recommendation_service.recommend(request)