"""Contrato comum aos backends de inferência."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from triage.data.schema import CONDITION_NAMES, LABELS
from triage.urgency import urgency_for


@dataclass(frozen=True)
class Prediction:
    """Resultado da triagem de um único laudo."""

    condition_label: int
    condition_name: str
    urgency: str
    confidence: float
    probabilities: dict[str, float]


@runtime_checkable
class Predictor(Protocol):
    """Interface implementada por todo backend de inferência."""

    backend: str
    metadata: dict

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica uma lista de laudos."""
        ...


def build_predictions(probability_matrix: np.ndarray) -> list[Prediction]:
    """Converte a matriz (n_amostras, 5) de probabilidades em Predictions.

    As colunas seguem a ordem de `LABELS`. `SklearnPredictor._reorder` impõe
    essa ordem ativamente; o backend ONNX não a reordena — ele apenas herda
    a ordem de `clf.classes_` no momento da conversão para ONNX, que já
    coincide com `LABELS` hoje, mas não é reforçada em tempo de inferência.
    """
    predictions = []
    for row in np.asarray(probability_matrix, dtype=float):
        index = int(np.argmax(row))
        label = LABELS[index]
        predictions.append(
            Prediction(
                condition_label=label,
                condition_name=CONDITION_NAMES[label],
                urgency=urgency_for(label),
                confidence=float(row[index]),
                probabilities={
                    CONDITION_NAMES[LABELS[i]]: float(value)
                    for i, value in enumerate(row)
                },
            )
        )
    return predictions


def ensure_non_empty(texts: list[str]) -> None:
    """Valida o lote de entrada."""
    if not texts:
        raise ValueError("a lista de textos não pode ser vazia")


def artifact_path(models_dir: Path, filename: str) -> Path:
    """Resolve o caminho de um artefato, levantando erro claro se faltar."""
    path = Path(models_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"artefato não encontrado: {path}")
    return path
