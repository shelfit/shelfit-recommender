from qdrant_client.http.models import models

from app.services.query_builder.abstract_query_filter import AbstractQueryFilter


class Range(AbstractQueryFilter):
    def __init__(self,
        key: str,
        gt: float|None = None,
        gte: float|None = None,
        lt: float|None = None,
        lte: float|None = None,
    ):
        self.key = key
        self.gt = gt
        self.gte = gte
        self.lt = lt
        self.lte = lte

    def get_field_condition(self) -> models.FieldCondition:
        return models.FieldCondition(
            key=self.key,
            range=models.Range(
                gt=self.gt,
                gte=self.gte,
                lt=self.lt,
                lte=self.lte,
            )
        )
