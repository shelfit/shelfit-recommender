from enum import StrEnum
from typing import List
from pydantic import BaseModel, ConfigDict


class RecommendRequest(BaseModel):
    query: str

class IntentItemType(StrEnum):
    AUTHOR = 'author'
    BOOK = 'book'
    GENRE = 'genre'

class IntentType(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    SIMILAR = "similar"

class IntentTerm(BaseModel):
    model_config = ConfigDict(extra='forbid')

    item: str
    item_type: IntentItemType
    intent: IntentType

class ParsedQueryIntent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    query_full: str
    query_context_residue: str
    terms: List[IntentTerm]