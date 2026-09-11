"""Construção dos pipelines candidatos (TF-IDF + classificador)."""

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from triage.config.seeds import SEED
from triage.features.text import build_vectorizer


def _linear_svc() -> ClassifierMixin:
    """LinearSVC calibrado — sem calibração não há predict_proba."""
    return CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=SEED),
        cv=3,
    )


CANDIDATE_MODELS: dict[str, Callable[[], ClassifierMixin]] = {
    "dummy": lambda: DummyClassifier(strategy="most_frequent"),
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
    ),
    "linear_svc": _linear_svc,
}


def build_pipeline(model_name: str, **vectorizer_kwargs) -> Pipeline:
    """Monta o pipeline TF-IDF + classificador para o modelo informado."""
    if model_name not in CANDIDATE_MODELS:
        raise ValueError(
            f"modelo desconhecido: {model_name!r}; "
            f"disponíveis: {sorted(CANDIDATE_MODELS)}"
        )
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**vectorizer_kwargs)),
            ("clf", CANDIDATE_MODELS[model_name]()),
        ]
    )
