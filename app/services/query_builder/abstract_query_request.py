from abc import ABC, abstractmethod

from qdrant_client.http.models import models


class AbstractQueryRequest(ABC):
    @abstractmethod
    def get_request(self) -> list[float] | models.Query:
        pass