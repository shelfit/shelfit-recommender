import sys

from app.models import RecommendRequest, FilterClause
from app.services.intent_parser import IntentParser
from app.services.qdrant_query_builder import QdrantQueryBuilder


class RecommendationService:
    MIN_NUM_RATINGS = 1000

    def __init__(self, intent_parser: IntentParser, qdrant_query_builder: QdrantQueryBuilder):
        self.intent_parser = intent_parser
        self.qdrant_query_builder = qdrant_query_builder

    def recommend(self, query: RecommendRequest):
        parsed_query_intent = self.intent_parser.parse(query.query)

        if parsed_query_intent is None:
            results = (
                self.qdrant_query_builder
                .query(query.query)
                .range(FilterClause.MUST, "num_ratings", gte=self.MIN_NUM_RATINGS)
                .execute()
            ).points

            results = sorted(results, key=lambda book: book.payload["rating"], reverse=True)
            return results[:3]
