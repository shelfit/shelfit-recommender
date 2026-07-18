from typing import Annotated

from fastapi import Depends
from openai import OpenAI

from app.core.config import settings
from app.services.intent_parser import IntentParser
from app.services.recommendation_service import RecommendationService


def get_intent_parser():
    client = OpenAI(api_key=settings.openrouter_api_key)
    return IntentParser(client)

def get_recommendation_service(intent_parser_service: Annotated[IntentParser, Depends(get_intent_parser)]):
    return RecommendationService(intent_parser_service)

IntentParserDep = Annotated[IntentParser, Depends(get_intent_parser)]
RecommendationServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]