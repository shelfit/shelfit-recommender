from typing import Self

from qdrant_client import models, QdrantClient
from qdrant_client.http.models import QueryResponse, FieldCondition, OrderBy

from app.core.config import settings
from app.models import FilterClause
from app.services.query_builder.abstract_query_filter import AbstractQueryFilter
from app.services.query_builder.abstract_query_request import AbstractQueryRequest


class QdrantQueryBuilder:
    DEFAULT_QUERY_LIMIT = 20
    DEFAULT_WITH_VECTORS = False
    DEFAULT_WITH_PAYLOAD = True

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client

        self._collection: str = settings.qdrant_collection_name
        self._query: AbstractQueryRequest|None = None
        self._query_filters: dict[FilterClause, list[FieldCondition]] = { clause: [] for clause in FilterClause }
        self._with_vectors: bool = self.DEFAULT_WITH_VECTORS
        self._with_payload: bool|list[str] = self.DEFAULT_WITH_PAYLOAD
        self._limit: int = self.DEFAULT_QUERY_LIMIT

    def query(self, query: AbstractQueryRequest)-> Self:
        self._query = query
        return self

    def add_must(self, query_filter: AbstractQueryFilter) -> Self:
        self._query_filters[FilterClause.MUST].append(query_filter.get_field_condition())
        return self

    def add_must_not(self, query_filter: AbstractQueryFilter) -> Self:
        self._query_filters[FilterClause.MUST_NOT].append(query_filter.get_field_condition())
        return self

    def with_vectors(self, with_vectors: bool)-> Self:
        self._with_vectors = with_vectors
        return self

    def with_payload(self, payload: bool|list[str])-> Self:
        self._with_payload = payload
        return self

    def limit(self, limit: int)-> Self:
        self._limit = limit
        return self

    def execute(self)-> QueryResponse:
        if self._query is None:
            raise RuntimeError("No query set in QdrantQueryBuilder")

        return self.qdrant_client.query_points(
            collection_name=self._collection,
            query=self._query.get_request(),
            query_filter=models.Filter(
                must=self._query_filters[FilterClause.MUST] if self._query_filters[FilterClause.MUST] else None,
                must_not=self._query_filters[FilterClause.MUST_NOT] if self._query_filters[FilterClause.MUST_NOT] else None,
                should=self._query_filters[FilterClause.SHOULD] if self._query_filters[FilterClause.SHOULD] else None,
            ),
            with_vectors=self._with_vectors,
            with_payload=self._with_payload,
            limit=self._limit,
        )
