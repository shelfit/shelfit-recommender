from app.models import RecommendRequest
from app.services.intent_parser import IntentParser


class RecommendationService:
    def __init__(self, intent_parser: IntentParser):
        self.intent_parser = intent_parser

    def recommend(self, query: RecommendRequest):
        return self.intent_parser.parse(query.query)