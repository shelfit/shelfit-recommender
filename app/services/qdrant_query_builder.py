from typing import Self

from qdrant_client import models, QdrantClient
from qdrant_client.http.models import QueryResponse, FieldCondition
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models import FilterClause


class QdrantQueryBuilder:
    DEFAULT_QUERY_LIMIT = 20

    def __init__(self, qdrant_client: QdrantClient, embedding_model: SentenceTransformer):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

        self._collection: str = settings.qdrant_collection_name
        self._query: list[float]|None = None
        self._filter_clauses: dict[FilterClause, list[FieldCondition]] = { clause: [] for clause in FilterClause }
        self._with_vectors: bool = False
        self._with_payload: bool|list[str] = True
        self._limit: int = self.DEFAULT_QUERY_LIMIT

    def query(self, query: str)-> Self:
        self._query = self.embedding_model.encode(query).tolist()
        return self

    def match(self, filter_clause: FilterClause, key: str, value: bool|int|str)-> Self:
        self._filter_clauses[filter_clause].append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
        )
        return self

    def range(self,
        filter_clause: FilterClause,
        key: str,
        gt: float|None = None,
        gte: float|None = None,
        lt: float|None = None,
        lte: float|None = None
    )-> Self:
        self._filter_clauses[filter_clause].append(
            models.FieldCondition(
                key=key,
                range=models.Range(
                    gt=gt,
                    gte=gte,
                    lt=lt,
                    lte=lte,
                )
            )
        )
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
            query=self._query,
            query_filter=models.Filter(
                must=self._filter_clauses[FilterClause.MUST] if self._filter_clauses[FilterClause.MUST] else None,
                must_not=self._filter_clauses[FilterClause.MUST_NOT] if self._filter_clauses[FilterClause.MUST_NOT] else None,
                should=self._filter_clauses[FilterClause.SHOULD] if self._filter_clauses[FilterClause.SHOULD] else None,
            ),
            with_vectors=self._with_vectors,
            with_payload=self._with_payload,
            limit=self._limit,
        )
