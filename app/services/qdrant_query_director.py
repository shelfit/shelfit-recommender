from typing import Self

from qdrant_client.http.models import ScoredPoint

from app.models import IntentItemType, IntentTerm, IntentType
from app.services.query_builder.match import Match
from app.services.query_builder.qdrant_query_builder import QdrantQueryBuilder
from app.services.qdrant_service import QdrantService
from app.services.query_builder.range import Range


class QdrantQueryDirector:
    NUM_RESULTS_RETURNED = 5
    MIN_NUM_RATINGS = 1000
    QUERY_TERM_TO_QDRANT_FIELD_MAP = {
        IntentItemType.BOOK: "title_normalized",
        IntentItemType.AUTHOR: "author_normalized",
        IntentItemType.GENRE: "genres_normalized",
    }

    def __init__(self, query_builder: QdrantQueryBuilder, qdrant_service: QdrantService):
        self.query_builder = query_builder
        self.qdrant_service = qdrant_service

    def query(self, query: str) -> Self:
        self.query_builder.query(query)
        return self

    def query_filters_from_terms(self, terms: list[IntentTerm]) -> Self:
        for term in terms:
            match term.intent:
                case IntentType.INCLUDE:
                    self._include(term)
                case IntentType.EXCLUDE:
                    self._exclude(term)
                case IntentType.SIMILAR:
                    pass
                case _:
                    continue

        return self

    def _include(self, term: IntentTerm) -> Self:
        self.query_builder.add_must(
            Match(
                self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type],
                term.item_normalized,
            )
        )
        return self

    def _exclude(self, term: IntentTerm) -> Self:
        self.query_builder.add_must_not(
            Match(
                self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type],
                term.item_normalized,
            )
        )
        return self

    def execute(self) -> list[ScoredPoint]:
        results = self.query_builder.execute()
        return results.points[:self.NUM_RESULTS_RETURNED]

    def similarity_search(self, query: str) -> list[ScoredPoint]:
        results = (self.query_builder
         .query(query)
         .add_must(Range("num_ratings", gte=self.MIN_NUM_RATINGS))
         .execute())
        return results.points[:self.NUM_RESULTS_RETURNED]
