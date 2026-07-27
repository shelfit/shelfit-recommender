from fastapi import FastAPI

from app.deps import RecommendationServiceDep
from app.models import RecommendRequest

app = FastAPI()

@app.post("/api/recommend")
def recommend(request: RecommendRequest, recommendation_service: RecommendationServiceDep):
    return recommendation_service.recommend(request)
