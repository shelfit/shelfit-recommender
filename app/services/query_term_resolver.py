from rapidfuzz import process, fuzz

from app.models import IntentTerm, IntentItemType
from app.services.vocabulary_store import VocabularyStore


class QueryTermResolver:
    ALIASES = {
        IntentItemType.GENRE: {
            "sci_fi": "science_fiction",
            "scifi": "science_fiction",
            "non_fiction": "nonfiction",
            "ya": "young_adult"
        }
    }

    SCORE_CUTOFF = 85
    MATCH_LIMIT = 50
    MATCH_LENGTH_CUTOFF = 0.6
    BEST_MATCH_CANDIDATE_WINDOW = 5

    def __init__(self, vocabulary_store: VocabularyStore):
        self._vocabulary_store = vocabulary_store

    def resolve(self, term: str, term_type: IntentItemType) -> str|None:
        term = self.ALIASES.get(term_type, {}).get(term, term)
        vocabulary = self._vocabulary_store.get_vocabularies()[term_type]

        if term in vocabulary.keys():
            return term

        matches = process.extract(
            term,
            vocabulary.keys(),
            scorer=fuzz.partial_ratio,
            score_cutoff=self.SCORE_CUTOFF,
            limit=self.MATCH_LIMIT,
        )
        matches = [match for match in matches if len(match[0]) >= len(term) * self.MATCH_LENGTH_CUTOFF]

        if not matches:
            return None

        best_match = matches[0][1]
        best_candidates = [match for match in matches if match[1] >= best_match - self.BEST_MATCH_CANDIDATE_WINDOW]
        return max(best_candidates, key=lambda candidate: vocabulary[candidate[0]])[0]
