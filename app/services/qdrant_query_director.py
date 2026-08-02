from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer

from app.models import IntentItemType, IntentTerm, IntentType, ParsedQueryIntent, IncludeQueryIntent, \
    ExcludeQueryIntent, SimilarQueryIntent, SortDirection
from app.services.query_builder.match import Match
from app.services.query_builder.qdrant_query_builder import QdrantQueryBuilder
from app.services.qdrant_service import QdrantService
from app.services.query_builder.range import Range
from app.services.query_builder.recommendation_search_request import RecommendationSearchRequest
from app.services.query_builder.similarity_search_request import SimilaritySearchRequest


class QdrantQueryDirector:
    MIN_NUM_RATINGS = 10000
    QUERY_TERM_TO_QDRANT_FIELD_MAP = {
        IntentItemType.BOOK: "title_normalized",
        IntentItemType.AUTHOR: "author_normalized",
        IntentItemType.GENRE: "genres_normalized",
    }

    SIMILARITY_SORT_WEIGHT = 0.7
    RATING_SORT_WEIGHT = 0.3

    def __init__(self, query_builder: QdrantQueryBuilder, qdrant_service: QdrantService, embedding_model: SentenceTransformer):
        self.query_builder = query_builder
        self.qdrant_service = qdrant_service
        self.embedding_model = embedding_model

    def recommendation_query(self, query_intent: ParsedQueryIntent) -> list[ScoredPoint]:
        user_query_str = query_intent.query_context_residue if query_intent.terms else query_intent.query_full
        user_query_encoded = self.embedding_model.encode(user_query_str).tolist()

        query_intent_list: list[IncludeQueryIntent | ExcludeQueryIntent | SimilarQueryIntent] = []
        for term in query_intent.terms:
            match term.intent:
                case IntentType.INCLUDE:
                    query_intent_list.append(self._include(term))
                case IntentType.EXCLUDE:
                    query_intent_list.append(self._exclude(term))
                case IntentType.SIMILAR:
                    if (intent := self._similar(term)) is not None:
                        query_intent_list.append(intent)

        request = None
        for intent in query_intent_list:
            match intent:
                case SimilarQueryIntent():
                    if request is None:
                        request = RecommendationSearchRequest(positive=[user_query_encoded])
                    request.add_positives(intent.positive).add_negatives(intent.negative)

                    if intent.key is not self.QUERY_TERM_TO_QDRANT_FIELD_MAP[IntentItemType.GENRE]:
                        self.query_builder.add_must_not(Match(key=intent.key, value=intent.value))
                case IncludeQueryIntent():
                    self.query_builder.add_must(Match(key=intent.key, value=intent.value))
                case ExcludeQueryIntent():
                    self.query_builder.add_must_not(Match(key=intent.key, value=intent.value))

        if request is None:
            request = SimilaritySearchRequest(query=user_query_encoded)

        results = (
            self.query_builder
            .query(request)
            .add_must(Range("num_ratings", gte=self.MIN_NUM_RATINGS))
            .execute()
        )
        return self._sort_points(results.points, SortDirection.DESC)


    def _include(self, term: IntentTerm) -> IncludeQueryIntent:
        return IncludeQueryIntent(
            key=self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type],
            value=term.item_normalized
        )

    def _exclude(self, term: IntentTerm) -> ExcludeQueryIntent:
        return ExcludeQueryIntent(
            key=self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type],
            value=term.item_normalized
        )

    def _similar(self, term: IntentTerm) -> SimilarQueryIntent|None:
        match term.item_type:
            case IntentItemType.BOOK:
                point = self.qdrant_service.fetch_book_by_title(term.item_normalized)

                if point is None:
                    return None
                positive = [point.vector]

            case IntentItemType.AUTHOR | IntentItemType.GENRE:
                field = self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type]
                points = self.qdrant_service.fetch_top_books_by_field(field, term.item_normalized)

                if not points:
                    return None
                positive = [point.vector for point in points]

        return SimilarQueryIntent(
            key=self.QUERY_TERM_TO_QDRANT_FIELD_MAP[term.item_type],
            value=term.item_normalized,
            positive=positive,
            negative=[]
        )

    def _sort_points(self, points: list[ScoredPoint], direction: SortDirection) -> list[ScoredPoint]:
        return sorted(
            points,
            key=lambda p: p.score * self.SIMILARITY_SORT_WEIGHT + (p.payload["rating"] / 5) * self.RATING_SORT_WEIGHT,
            reverse=direction is SortDirection.DESC
        )

    def similarity_search(self, query: str) -> list[ScoredPoint]:
        query_encoded = self.embedding_model.encode(query).tolist()

        results = (self.query_builder
            .query(SimilaritySearchRequest(query_encoded))
            .add_must(Range("num_ratings", gte=self.MIN_NUM_RATINGS))
            .execute())

        return self._sort_points(results.points, SortDirection.DESC)
