from qdrant_client import QdrantClient
from qdrant_client.http.models import models

from app.core.config import settings


class QdrantService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def fetch_top_books_by_field(self, key: str, value: str, limit: int = 3) -> list[models.Record]:
        return self.client.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ),
                ]
            ),
            order_by=models.OrderBy(
                key="rating",
                direction=models.Direction.DESC
            ),
            limit=limit,
            with_vectors=True,
            with_payload=True,
        )[0]

    def fetch_book_by_title(self, title: str) -> models.Record | None:
        result = self.client.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="title_normalized",
                        match=models.MatchValue(value=title)
                    )
                ]
            ),
            limit=1,
            with_vectors=True,
            with_payload=True,
        )[0]

        if not result:
            return None

        return result[0]
