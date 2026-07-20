import unicodedata


class StringFormatterUtil:
    # characters with no NFKD decomposition, so they survive the combining-mark
    # strip below and have to be mapped by hand
    _CHAR_FOLDS = str.maketrans({
        "ø": "o", "đ": "d", "ð": "d", "ł": "l", "ħ": "h",
        "æ": "ae", "œ": "oe", "þ": "th", "ß": "ss",
    })

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
        # this is to format names like j.r.r. tolkien to jrr_tolkien instead of j.r.r_tolkien
        if "." in term:
            term = term.replace(".", "")

        if "'" in term:
            term = term.replace("'", "")

        # brontë -> bronte, so accented source data matches unaccented queries
        term = term.lower().translate(StringFormatterUtil._CHAR_FOLDS)
        term = unicodedata.normalize("NFKD", term)
        term = "".join(c for c in term if not unicodedata.combining(c))

        return "_".join(term.split())