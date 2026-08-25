import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.deps import RecommendationServiceDep, get_vocabulary_store, QueryDirectorDep, \
    get_embedding_model, get_reranking_model
from app.models import RecommendRequest, SearchRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_vocabulary_store().refresh()
    except Exception as e:
        logging.error(f"Error in lifespan: {str(e)}")

    try:
        _warm_up_models()
    except Exception as e:
        logging.error(f"Error warming up models: {str(e)}")

    yield


def _warm_up_models():
    get_embedding_model().encode("warm up")
    get_reranking_model().predict([("warm up", "warm up")])

app = FastAPI(lifespan=lifespan)

@app.post("/api/recommend")
def recommend(request: RecommendRequest, recommendation_service: RecommendationServiceDep):
    return recommendation_service.recommend(request)

@app.post("/api/search")
def search(request: SearchRequest, query_director: QueryDirectorDep):
    results = query_director.similarity_search(request.query)
    result_ids = [point.payload['id'] for point in results]
    return result_ids[request.offset:(request.offset + request.limit)]
