from app.models import RecommendRequest, IntentType
from app.services.intent_parser import IntentParser
from app.services.qdrant_query_director import QdrantQueryDirector


class RecommendationService:
    NUM_RESULTS_RECOMMENDED = 5

    def __init__(self, intent_parser: IntentParser, qdrant_query_director: QdrantQueryDirector):
        self.intent_parser = intent_parser
        self.qdrant_query_director = qdrant_query_director

    def recommend(self, query: RecommendRequest):
        parsed_query_intent = self.intent_parser.parse(query.query)

        if parsed_query_intent is None:
            results = self.qdrant_query_director.similarity_search(query.query)
        else:
            results = self.qdrant_query_director.recommendation_query(parsed_query_intent)

        results_ids = [point.payload["id"] for point in results]
        return results_ids[:self.NUM_RESULTS_RECOMMENDED]
