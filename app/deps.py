from typing import Annotated

from fastapi import Depends
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.intent_parser import IntentParser
from app.services.query_builder.qdrant_query_builder import QdrantQueryBuilder
from app.services.qdrant_query_director import QdrantQueryDirector
from app.services.qdrant_service import QdrantService
from app.services.recommendation_service import RecommendationService

openai_client = OpenAI(api_key=settings.openrouter_api_key)
qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
embedding_model = SentenceTransformer(settings.embedding_model)

def get_openai_client()-> OpenAI:
    return openai_client

def get_qdrant_client()-> QdrantClient:
    return qdrant_client

def get_embedding_model() -> SentenceTransformer:
    return embedding_model

def get_intent_parser(client: Annotated[OpenAI, Depends(get_openai_client)])-> IntentParser:
    return IntentParser(client)

def get_qdrant_service(client: Annotated[QdrantClient, Depends(get_qdrant_client)])-> QdrantService:
    return QdrantService(client)

def get_qdrant_query_builder(
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
    model: Annotated[SentenceTransformer, Depends(get_embedding_model)]
)-> QdrantQueryBuilder:
    return QdrantQueryBuilder(client, model)

def get_qdrant_query_director(
    query_builder: Annotated[QdrantQueryBuilder, Depends(get_qdrant_query_builder)],
    qdrant_service: Annotated[QdrantService, Depends(get_qdrant_service)]
) -> QdrantQueryDirector:
    return QdrantQueryDirector(query_builder, qdrant_service)

def get_recommendation_service(
    intent_parser: Annotated[IntentParser, Depends(get_intent_parser)],
    qdrant_query_director: Annotated[QdrantQueryDirector, Depends(get_qdrant_query_director)]
)-> RecommendationService:
    return RecommendationService(intent_parser, qdrant_query_director)

IntentParserDep = Annotated[IntentParser, Depends(get_intent_parser)]
RecommendationServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]