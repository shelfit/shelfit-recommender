from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.models import RecommendRequest, Settings

settings = Settings() # noqa

model = SentenceTransformer(settings.embedding_model)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

app = FastAPI()

@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    embedding = model.encode(request.query)

    search_result = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=embedding.tolist(),
        limit=10
    )

    return search_result