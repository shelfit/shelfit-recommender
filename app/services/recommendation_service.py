from app.models import RecommendRequest
from app.services.intent_parser_service import IntentParserService


class RecommendationService:
    def __init__(self, intent_parser: IntentParserService):
        self.intent_parser = intent_parser

    def recommend(self, query: RecommendRequest):
        return self.intent_parser.parse(query.query)