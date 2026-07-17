from qdrant_client import QdrantClient
from app.models import ParsedQueryIntent


class QdrantService:
    def __init__(self, client: QdrantClient):
        self.client = client

        