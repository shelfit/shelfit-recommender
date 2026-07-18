from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.deps import RecommendationServiceDep
from app.models import RecommendRequest, ParsedQueryIntent

app = FastAPI()

@app.post("/api/recommend")
def recommend(
    request: RecommendRequest,
):
    model = SentenceTransformer(settings.embedding_model)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    embedding = model.encode(request.query)

    search_result = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=embedding.tolist(),
        limit=10
    )

    return search_result

@app.post("/api/intent-parse")
def intent_parse(
    request: RecommendRequest,
    recommendation_service: RecommendationServiceDep,
):
    return recommendation_service.recommend(request)