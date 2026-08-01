from typing import Self

from qdrant_client.http.models import models

from app.services.query_builder.abstract_query_request import AbstractQueryRequest


class RecommendationSearchRequest(AbstractQueryRequest):
    def __init__(self,
        positive: list[int|list[float]] | None = None,
        negative: list[int|list[float]] | None = None,
        strategy: models.RecommendStrategy | None = None
    ):
        self._positive: list[int|list[float]] = positive if positive is not None else []
        self._negative: list[int|list[float]] = negative if negative is not None else []
        self._strategy: models.RecommendStrategy = strategy or models.RecommendStrategy.AVERAGE_VECTOR

    def add_positive(self, value: int | list[float]) -> Self:
        self._positive.append(value)
        return self

    def add_positives(self, values: list[int | list[float]]) -> Self:
        self._positive.extend(values)
        return self

    def add_negative(self, value: int | list[float]) -> Self:
        self._negative.append(value)
        return self

    def add_negatives(self, values: list[int | list[float]]) -> Self:
        self._negative.extend(values)
        return self

    def set_strategy(self, strategy: models.RecommendStrategy) -> Self:
        self._strategy = strategy
        return self

    def get_request(self) -> models.RecommendQuery:
        return models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=self._positive,
                negative=self._negative,
                strategy=self._strategy
            )
        )