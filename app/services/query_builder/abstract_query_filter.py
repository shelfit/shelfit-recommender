from abc import ABC, abstractmethod

from qdrant_client.http.models import FieldCondition


class AbstractQueryFilter(ABC):
    @classmethod
    @abstractmethod
    def get_field_condition(cls) -> FieldCondition:
        pass
