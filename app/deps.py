from functools import lru_cache
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
from app.services.query_term_resolver import QueryTermResolver
from app.services.recommendation_service import RecommendationService
from app.services.vocabulary_store import VocabularyStore


@lru_cache
def get_openai_client()-> OpenAI:
    return OpenAI(api_key=settings.openrouter_api_key)

@lru_cache
def get_qdrant_client()-> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

@lru_cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)

@lru_cache
def get_vocabulary_store() -> VocabularyStore:
    return VocabularyStore(get_qdrant_client())

def get_term_resolver(vocabulary_store: Annotated[VocabularyStore, Depends(get_vocabulary_store)]) -> QueryTermResolver:
    return QueryTermResolver(vocabulary_store)

def get_intent_parser(
    client: Annotated[OpenAI, Depends(get_openai_client)],
    term_resolver: Annotated[QueryTermResolver, Depends(get_term_resolver)]
)-> IntentParser:
    return IntentParser(client, term_resolver)

def get_qdrant_service(client: Annotated[QdrantClient, Depends(get_qdrant_client)])-> QdrantService:
    return QdrantService(client)

def get_qdrant_query_builder(client: Annotated[QdrantClient, Depends(get_qdrant_client)])-> QdrantQueryBuilder:
    return QdrantQueryBuilder(client)

def get_qdrant_query_director(
    query_builder: Annotated[QdrantQueryBuilder, Depends(get_qdrant_query_builder)],
    qdrant_service: Annotated[QdrantService, Depends(get_qdrant_service)],
    model: Annotated[SentenceTransformer, Depends(get_embedding_model)]
) -> QdrantQueryDirector:
    return QdrantQueryDirector(query_builder, qdrant_service, model)

def get_recommendation_service(
    intent_parser: Annotated[IntentParser, Depends(get_intent_parser)],
    qdrant_query_director: Annotated[QdrantQueryDirector, Depends(get_qdrant_query_director)]
)-> RecommendationService:
    return RecommendationService(intent_parser, qdrant_query_director)

IntentParserDep = Annotated[IntentParser, Depends(get_intent_parser)]
RecommendationServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]