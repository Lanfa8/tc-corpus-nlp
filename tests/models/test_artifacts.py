import pytest

from triage.models.artifacts import load_pipeline, save_pipeline
from triage.models.factory import build_pipeline


def test_roundtrip_preserves_predictions(tmp_path, synthetic_frame):
    pipeline = build_pipeline("logistic_regression", min_df=1)
    texts = synthetic_frame["medical_abstract"].tolist()
    pipeline.fit(texts, synthetic_frame["condition_label"].tolist())
    expected = pipeline.predict(texts[:10]).tolist()

    save_pipeline(pipeline, {"model_name": "logistic_regression"}, tmp_path)
    loaded, metadata = load_pipeline(tmp_path)

    assert loaded.predict(texts[:10]).tolist() == expected
    assert metadata["model_name"] == "logistic_regression"


def test_load_raises_when_artifacts_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path)
