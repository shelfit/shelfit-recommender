from typing import Self

from qdrant_client.http.models import ScoredPoint

from app.models import FilterClause, IntentItemType, IntentTerm
from app.services.qdrant_query_builder import QdrantQueryBuilder


class QdrantQueryDirector:
    NUM_RESULTS_RETURNED = 5
    MIN_NUM_RATINGS = 1000
    QUERY_TERM_TO_QDRANT_FIELD_MAP = {
        IntentItemType.BOOK: "title_normalized",
        IntentItemType.AUTHOR: "author_normalized",
        IntentItemType.GENRE: "genres_normalized",
    }

    def __init__(self, query_builder: QdrantQueryBuilder):
        self.query_builder = query_builder

    def query(self, query: str) -> Self:
        self.query_builder.query(query)
        return self

    def include(self, term: IntentTerm) -> Self:
        self.query_builder.match(FilterClause.MUST, self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type], term.item_normalized)
        return self

    def exclude(self, term: IntentTerm) -> Self:
        self.query_builder.match(FilterClause.MUST_NOT, self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type], term.item_normalized)
        return self

    def execute(self) -> list[ScoredPoint]:
        results = self.query_builder.execute()
        return results.points[:self.NUM_RESULTS_RETURNED]

    def similarity_search(self, query: str) -> list[ScoredPoint]:
        results = (self.query_builder
         .query(query)
         .range(FilterClause.MUST, "num_ratings", gte=self.MIN_NUM_RATINGS)
         .execute())
        return results.points[:self.NUM_RESULTS_RETURNED]
