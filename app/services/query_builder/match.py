from qdrant_client import models

from app.services.query_builder.abstract_query_filter import AbstractQueryFilter


class Match(AbstractQueryFilter):
    def __init__(self, key: str, value: int|str|bool):
        self.key = key
        self.value = value

    def get_field_condition(self) -> models.FieldCondition:
        return models.FieldCondition(
            key=self.key,
            match=models.MatchValue(value=self.value)
        )
