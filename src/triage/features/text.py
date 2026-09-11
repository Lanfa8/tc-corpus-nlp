"""Normalização de texto e construção do vetorizador TF-IDF."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer

_NON_LETTER = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Minúsculas, remove dígitos e pontuação, colapsa espaços."""
    lowered = str(text).lower()
    letters_only = _NON_LETTER.sub(" ", lowered)
    return _WHITESPACE.sub(" ", letters_only).strip()


def build_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int = 50_000,
) -> TfidfVectorizer:
    """Cria o TfidfVectorizer usado pelo pipeline de treino e de inferência."""
    return TfidfVectorizer(
        preprocessor=normalize_text,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
