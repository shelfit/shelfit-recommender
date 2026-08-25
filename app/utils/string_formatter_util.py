import re
import unicodedata

from qdrant_client.http.models import ScoredPoint


class StringFormatterUtil:
    # characters with no NFKD decomposition, so they survive the combining-mark
    # strip below and have to be mapped by hand
    _CHAR_FOLDS = str.maketrans({
        "ø": "o", "đ": "d", "ð": "d", "ł": "l", "ħ": "h",
        "æ": "ae", "œ": "oe", "þ": "th", "ß": "ss",
    })
    _SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s]")

    _RERANKING_STRING_FORMAT = "{title} by {author}. {description}"

    @staticmethod
    def remove_json_markdown(model_response: str)-> str:
        if model_response.startswith('```'):
            model_response = model_response[3:]

            if model_response[:4].lower() == 'json':
                model_response = model_response[4:]

            model_response = model_response[:-3]

        return model_response

    # applies normalization rules to author names and book titles
    # during embedding and search
    @staticmethod
    def normalize_term(term: str) -> str:
        # vocabulary-parsed terms get an underscore, replace with a space
        # so it can be returned to an underscore later
        term = term.replace("_", " ")

        # brontë -> bronte, so accented source data matches unaccented queries
        term = term.lower().translate(StringFormatterUtil._CHAR_FOLDS)
        term = unicodedata.normalize("NFKD", term)
        term = "".join(c for c in term if not unicodedata.combining(c))

        # remove every special character (anything that isn't a letter, digit, or whitespace)
        term = StringFormatterUtil._SPECIAL_CHARS.sub("", term)

        return "_".join(term.split())

    @staticmethod
    def format_reranking_string(point: ScoredPoint) -> str:
        return StringFormatterUtil._RERANKING_STRING_FORMAT.format(
            title=point.payload["title"],
            author=point.payload["author"],
            description=point.payload["description"]
        )
