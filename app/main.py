from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.models import RecommendRequest

app = FastAPI()
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    embedding = model.encode(request.query)

    client = QdrantClient('qdrant', port=6333)
    search_result = client.query_points(
        collection_name='books',
        query=embedding.tolist(),
        limit=10
    )

    return search_result