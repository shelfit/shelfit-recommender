from qdrant_client import QdrantClient


class QdrantService:
    def __init__(self, client: QdrantClient):
        self.client = client

        