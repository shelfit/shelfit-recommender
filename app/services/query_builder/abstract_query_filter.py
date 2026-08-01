from abc import ABC, abstractmethod

from qdrant_client.http.models import FieldCondition


class AbstractQueryFilter(ABC):
    @abstractmethod
    def get_field_condition(self) -> FieldCondition:
        pass
