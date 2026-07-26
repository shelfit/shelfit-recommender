from app.models import RecommendRequest, IntentType
from app.services.intent_parser import IntentParser
from app.services.qdrant_query_director import QdrantQueryDirector


class RecommendationService:
    def __init__(self, intent_parser: IntentParser, qdrant_query_director: QdrantQueryDirector):
        self.intent_parser = intent_parser
        self.qdrant_query_director = qdrant_query_director

    def recommend(self, query: RecommendRequest):
        parsed_query_intent = self.intent_parser.parse(query.query)

        if parsed_query_intent is None:
            return self.qdrant_query_director.similarity_search(query.query)

        self.qdrant_query_director.query(
            parsed_query_intent.query_context_residue if parsed_query_intent.terms else parsed_query_intent.query_full
        )

        for term in parsed_query_intent.terms:
            match term.intent:
                case IntentType.INCLUDE:
                    self.qdrant_query_director.include(term)
                case IntentType.EXCLUDE:
                    self.qdrant_query_director.exclude(term)
                case IntentType.SIMILAR:
                    pass
                case _:
                    continue

        return self.qdrant_query_director.execute()
