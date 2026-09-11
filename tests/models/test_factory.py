import pytest
from sklearn.pipeline import Pipeline

from triage.models.factory import CANDIDATE_MODELS, build_pipeline


def test_candidate_models_are_registered():
    assert set(CANDIDATE_MODELS) == {
        "dummy",
        "naive_bayes",
        "logistic_regression",
        "linear_svc",
    }


@pytest.mark.parametrize("model_name", sorted(CANDIDATE_MODELS))
def test_build_pipeline_has_named_steps(model_name):
    pipeline = build_pipeline(model_name)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["tfidf", "clf"]


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="desconhecido"):
        build_pipeline("random_forest")


def test_all_candidates_expose_predict_proba(synthetic_frame):
    """LinearSVC não tem predict_proba nativo; a factory o calibra."""
    for model_name in CANDIDATE_MODELS:
        pipeline = build_pipeline(model_name, min_df=1)
        pipeline.fit(
            synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
        )
        probabilities = pipeline.predict_proba(["tumor malignant growth"])
        assert probabilities.shape == (1, 5)
