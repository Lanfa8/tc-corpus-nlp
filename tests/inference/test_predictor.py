import pytest

from triage.inference import load_predictor
from triage.inference.base import Prediction
from triage.inference.predictor import SklearnPredictor
from triage.models.artifacts import save_pipeline
from triage.models.factory import build_pipeline


@pytest.fixture
def artifacts_dir(tmp_path, synthetic_frame):
    pipeline = build_pipeline("logistic_regression", min_df=1)
    pipeline.fit(
        synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
    )
    save_pipeline(pipeline, {"model_name": "logistic_regression"}, tmp_path)
    return tmp_path


def test_predict_returns_one_prediction_per_text(artifacts_dir):
    predictor = SklearnPredictor.from_artifacts(artifacts_dir)
    results = predictor.predict(["tumor malignant growth", "cardiac heart arterial"])
    assert len(results) == 2
    assert all(isinstance(r, Prediction) for r in results)


def test_prediction_fields_are_coherent(artifacts_dir):
    predictor = SklearnPredictor.from_artifacts(artifacts_dir)
    result = predictor.predict(["tumor malignant carcinoma metastasis"])[0]

    assert result.condition_label == 1
    assert result.condition_name == "neoplasms"
    assert result.urgency == "urgente"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 5
    assert result.probabilities["neoplasms"] == pytest.approx(result.confidence)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_backend_name_is_sklearn(artifacts_dir):
    assert SklearnPredictor.from_artifacts(artifacts_dir).backend == "sklearn"


def test_predict_rejects_empty_batch(artifacts_dir):
    predictor = SklearnPredictor.from_artifacts(artifacts_dir)
    with pytest.raises(ValueError, match="vazia"):
        predictor.predict([])


def test_load_predictor_resolves_sklearn_backend(artifacts_dir):
    predictor = load_predictor("sklearn", artifacts_dir)
    assert isinstance(predictor, SklearnPredictor)


def test_load_predictor_rejects_unknown_backend(artifacts_dir):
    with pytest.raises(ValueError, match="backend"):
        load_predictor("tensorrt", artifacts_dir)
