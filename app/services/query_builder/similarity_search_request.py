from typing import Self

from app.services.query_builder.abstract_query_request import AbstractQueryRequest


class SimilaritySearchRequest(AbstractQueryRequest):
    def __init__(self, query: list[float] | None = None):
        self._query: list[float] = query if query is not None else []

    def set_query(self, query: list[float]) -> Self:
        self._query = query
        return self

    def get_request(self) -> list[float]:
        return self._query