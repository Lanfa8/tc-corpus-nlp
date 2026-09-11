import subprocess
import sys
from dataclasses import dataclass

import pytest

from triage.inference import load_predictor
from triage.inference.onnx_predictor import OnnxPredictor, _probabilities_output_name
from triage.inference.predictor import SklearnPredictor
from triage.models.artifacts import save_pipeline
from triage.models.factory import build_pipeline


@dataclass
class _FakeOutput:
    name: str
    shape: list


class _FakeSession:
    def __init__(self, outputs: list[_FakeOutput]) -> None:
        self._outputs = outputs

    def get_outputs(self) -> list[_FakeOutput]:
        return self._outputs


SAMPLE_TEXTS = [
    "tumor malignant carcinoma metastasis oncology growth",
    "gastric intestinal bowel liver hepatic digestion ulcer",
    "neural brain seizure cerebral cognitive neuropathy nerve",
    "cardiac heart arterial coronary myocardial vascular pressure",
    "inflammation infection fever chronic general pathology syndrome",
]


@pytest.fixture
def exported_dir(tmp_path, synthetic_frame):
    """Treina, salva e exporta o pipeline para ONNX via o script real."""
    pipeline = build_pipeline("logistic_regression", min_df=1)
    pipeline.fit(
        synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
    )
    save_pipeline(pipeline, {"model_name": "logistic_regression"}, tmp_path)

    result = subprocess.run(
        [sys.executable, "scripts/export_onnx.py", "--models-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pipeline.onnx").exists()
    return tmp_path


def test_probabilities_output_name_resolves_real_exported_graph(exported_dir):
    onnx_predictor = OnnxPredictor.from_artifacts(exported_dir)
    assert onnx_predictor._output_name == "probabilities"


def test_probabilities_output_name_rejects_wrong_output_count():
    session = _FakeSession([_FakeOutput("only_one", [None, 5])])
    with pytest.raises(ValueError, match="2 saídas"):
        _probabilities_output_name(session)


def test_probabilities_output_name_rejects_unrecognizable_shapes():
    session = _FakeSession(
        [_FakeOutput("label", [None]), _FakeOutput("mystery", [None, 3])]
    )
    with pytest.raises(ValueError, match="não foi possível identificar"):
        _probabilities_output_name(session)


def test_probabilities_output_name_falls_back_to_shape_when_unnamed():
    session = _FakeSession(
        [_FakeOutput("label", [None]), _FakeOutput("output_1", [None, 5])]
    )
    assert _probabilities_output_name(session) == "output_1"


def test_onnx_backend_name(exported_dir):
    assert OnnxPredictor.from_artifacts(exported_dir).backend == "onnx"


def test_onnx_predicts_same_classes_as_sklearn(exported_dir):
    sklearn_predictor = SklearnPredictor.from_artifacts(exported_dir)
    onnx_predictor = OnnxPredictor.from_artifacts(exported_dir)

    sklearn_labels = [
        p.condition_label for p in sklearn_predictor.predict(SAMPLE_TEXTS)
    ]
    onnx_labels = [p.condition_label for p in onnx_predictor.predict(SAMPLE_TEXTS)]

    assert sklearn_labels == onnx_labels


def test_onnx_probabilities_match_sklearn(exported_dir):
    sklearn_predictor = SklearnPredictor.from_artifacts(exported_dir)
    onnx_predictor = OnnxPredictor.from_artifacts(exported_dir)

    for expected, actual in zip(
        sklearn_predictor.predict(SAMPLE_TEXTS),
        onnx_predictor.predict(SAMPLE_TEXTS),
        strict=True,
    ):
        for condition, probability in expected.probabilities.items():
            assert actual.probabilities[condition] == pytest.approx(
                probability, abs=1e-4
            )


def test_onnx_prediction_carries_urgency(exported_dir):
    result = OnnxPredictor.from_artifacts(exported_dir).predict(
        ["tumor malignant carcinoma metastasis"]
    )[0]
    assert result.condition_name == "neoplasms"
    assert result.urgency == "urgente"


def test_onnx_rejects_empty_batch(exported_dir):
    with pytest.raises(ValueError, match="vazia"):
        OnnxPredictor.from_artifacts(exported_dir).predict([])


def test_load_predictor_resolves_onnx_backend(exported_dir):
    assert isinstance(load_predictor("onnx", exported_dir), OnnxPredictor)


def test_onnx_predictor_fails_clearly_without_artifact(tmp_path, synthetic_frame):
    pipeline = build_pipeline("logistic_regression", min_df=1)
    pipeline.fit(
        synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
    )
    save_pipeline(pipeline, {}, tmp_path)  # joblib existe, onnx não
    with pytest.raises(FileNotFoundError, match="pipeline.onnx"):
        OnnxPredictor.from_artifacts(tmp_path)
