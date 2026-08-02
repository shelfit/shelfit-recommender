import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.deps import RecommendationServiceDep, get_vocabulary_store
from app.models import RecommendRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_vocabulary_store().refresh()
    except Exception as e:
        logging.error(f"Error in lifespan: {str(e)}")

    yield

app = FastAPI(lifespan=lifespan)

@app.post("/api/recommend")
def recommend(request: RecommendRequest, recommendation_service: RecommendationServiceDep):
    return recommendation_service.recommend(request)
