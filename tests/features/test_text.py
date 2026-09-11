import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from triage.features.text import build_vectorizer, normalize_text


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_text("  Tissue   CHANGES\naround ") == "tissue changes around"


def test_normalize_strips_digits_and_punctuation():
    assert normalize_text("IL-1 levels (n=7) rose 25%.") == "il levels n rose"


def test_normalize_is_idempotent():
    once = normalize_text("Loose Prostheses -- 3 dogs!")
    assert normalize_text(once) == once


def test_normalize_handles_empty_input():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_build_vectorizer_uses_normalizer_as_preprocessor():
    vectorizer = build_vectorizer()
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.preprocessor is None
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.sublinear_tf is True


@pytest.mark.parametrize(
    "raw",
    [
        "IL-1 levels (n=7) rose 25% in the Cardiac Tissue.",
        "  Tissue   CHANGES\naround the Neural Growth-Plate ",
        "Loose Prostheses -- 3 dogs! Malignant carcinoma, metastasis?",
        "",
    ],
)
def test_vectorizer_analyzer_matches_normalize_text_oracle(raw):
    """token_pattern sobre o texto bruto deve tokenizar igual a normalize_text.

    Isso é o que torna a mudança do preprocessor customizado para
    token_pattern (necessária para exportar para ONNX) comportamentalmente
    equivalente: tokenizar o texto bruto com o padrão de tokens direto deve
    produzir exatamente os mesmos tokens que tokenizar a saída de
    `normalize_text` sobre o mesmo texto.
    """
    vectorizer = build_vectorizer()
    analyzer = vectorizer.build_analyzer()

    tokens_from_raw = analyzer(raw)
    tokens_from_normalized = analyzer(normalize_text(raw))

    assert tokens_from_raw == tokens_from_normalized


def test_vectorizer_fits_and_transforms():
    vectorizer = build_vectorizer(min_df=1)
    matrix = vectorizer.fit_transform(
        ["tumor growth in the liver", "cardiac arrest and chest pain"]
    )
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0
