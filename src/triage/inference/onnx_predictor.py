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
EXPECTED_OUTPUT_NAME = "probabilities"
EXPECTED_N_CLASSES = 5


def _probabilities_output_name(session: ort.InferenceSession) -> str:
    """Localiza a saída de probabilidades do grafo, sem confiar em posição.

    O grafo exportado (`zipmap=False`) tem duas saídas: `label` (rótulo) e
    `probabilities` (matriz `(n, 5)`). Ler `get_outputs()[1]` por posição, sem
    validar nada, faria uma reordenação futura do conversor (ou uma troca de
    versão do skl2onnx) alimentar rótulos inteiros como se fossem
    probabilidades, silenciosamente. Aceita a saída cujo nome seja o esperado
    ou, na ausência dele, a única saída 2-D de 5 colunas.
    """
    outputs = session.get_outputs()
    if len(outputs) != 2:
        raise ValueError(
            f"grafo ONNX inesperado: esperadas 2 saídas, encontradas {len(outputs)}"
        )

    by_name = next((o for o in outputs if o.name == EXPECTED_OUTPUT_NAME), None)
    if by_name is not None:
        return by_name.name

    candidates = [o for o in outputs if len(o.shape) == 2 and o.shape[-1] == 5]
    if len(candidates) != 1:
        raise ValueError(
            "não foi possível identificar a saída de probabilidades do grafo "
            f"ONNX: nenhuma saída chamada {EXPECTED_OUTPUT_NAME!r} e "
            f"{len(candidates)} saída(s) 2-D com {EXPECTED_N_CLASSES} colunas "
            f"(esperada exatamente 1); saídas disponíveis: "
            f"{[(o.name, o.shape) for o in outputs]}"
        )
    return candidates[0].name


class OnnxPredictor:
    """Inferência via ONNX Runtime — o pipeline inteiro roda como um grafo."""

    backend = "onnx"

    def __init__(self, session: ort.InferenceSession, metadata: dict) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        # A segunda saída do grafo é a matriz de probabilidades (zipmap desativado).
        self._output_name = _probabilities_output_name(session)
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
