from qdrant_client import QdrantClient

from app.core.config import settings
from app.models import IntentItemType


class VocabularyStore:
    AUTHOR_LIMIT = 30_000
    TITLE_LIMIT = 75_000
    GENRE_LIMIT = 3_000

    def __init__(self, client: QdrantClient):
        self._client = client

        self._authors_vocabulary: dict[str, int] = {}
        self._titles_vocabulary: dict[str, int] = {}
        self._genres_vocabulary: dict[str, int] = {}

    def refresh(self):
        author_hits = self._client.facet(
            collection_name=settings.qdrant_collection_name,
            key="author_normalized",
            limit=self.AUTHOR_LIMIT,
            exact=False,
        ).hits
        title_hits = self._client.facet(
            collection_name=settings.qdrant_collection_name,
            key="title_normalized",
            limit=self.TITLE_LIMIT,
            exact=False,
        ).hits
        genres_hits = self._client.facet(
            collection_name=settings.qdrant_collection_name,
            key="genres_normalized",
            limit=self.GENRE_LIMIT,
            exact=False,
        ).hits

        self._authors_vocabulary = { str(hit.value): hit.count for hit in author_hits }
        self._titles_vocabulary = { str(hit.value): hit.count for hit in title_hits }
        self._genres_vocabulary = { str(hit.value): hit.count for hit in genres_hits }

    def get_vocabularies(self) -> dict[str, dict[str, int]]:
        return {
            IntentItemType.AUTHOR: self._authors_vocabulary,
            IntentItemType.BOOK: self._titles_vocabulary,
            IntentItemType.GENRE: self._genres_vocabulary,
        }
