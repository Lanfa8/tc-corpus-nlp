"""Backend de inferência baseado no ONNX Runtime."""

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from triage.inference.base import (
    Prediction,
    artifact_path,
    build_predictions,
    ensure_non_empty,
)
from triage.models.artifacts import METADATA_FILE

logger = logging.getLogger(__name__)

ONNX_FILE = "pipeline.onnx"


class OnnxPredictor:
    """Inferência via ONNX Runtime — o pipeline inteiro roda como um grafo."""

    backend = "onnx"

    def __init__(self, session: ort.InferenceSession, metadata: dict) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        # A segunda saída do grafo é a matriz de probabilidades (zipmap desativado).
        self._output_name = session.get_outputs()[1].name
        self.metadata = metadata

    @classmethod
    def from_artifacts(cls, models_dir: Path) -> "OnnxPredictor":
        models_dir = Path(models_dir)
        onnx_path = artifact_path(models_dir, ONNX_FILE)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1  # evita contenção sob concorrência da API

        session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

        metadata_path = models_dir / METADATA_FILE
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        logger.info("backend onnx carregado de %s", onnx_path)
        return cls(session, metadata)

    def predict(self, texts: list[str]) -> list[Prediction]:
        ensure_non_empty(texts)
        inputs = np.array(texts, dtype=object).reshape(-1, 1)
        probabilities = self._session.run(
            [self._output_name], {self._input_name: inputs}
        )[0]
        return build_predictions(probabilities)
