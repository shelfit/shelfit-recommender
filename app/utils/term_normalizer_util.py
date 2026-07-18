class TermNormalizerUtil:
    @staticmethod
    def normalize(term: str)-> str:
        return "_".join(term.lower().split())