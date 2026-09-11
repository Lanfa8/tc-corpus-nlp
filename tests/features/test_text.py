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
    assert vectorizer.preprocessor is normalize_text
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.sublinear_tf is True


def test_vectorizer_fits_and_transforms():
    vectorizer = build_vectorizer(min_df=1)
    matrix = vectorizer.fit_transform(
        ["tumor growth in the liver", "cardiac arrest and chest pain"]
    )
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0
