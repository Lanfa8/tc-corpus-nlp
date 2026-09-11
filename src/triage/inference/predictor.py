"""Backend de inferência baseado no pipeline sklearn."""

import logging
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from triage.data.schema import LABELS
from triage.inference.base import Prediction, build_predictions, ensure_non_empty
from triage.models.artifacts import load_pipeline

logger = logging.getLogger(__name__)


class SklearnPredictor:
    """Inferência via `Pipeline` sklearn carregado do joblib."""

    backend = "sklearn"

    def __init__(self, pipeline: Pipeline, metadata: dict) -> None:
        self._pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def from_artifacts(cls, models_dir: Path) -> "SklearnPredictor":
        pipeline, metadata = load_pipeline(models_dir)
        logger.info("backend sklearn carregado de %s", models_dir)
        return cls(pipeline, metadata)

    def predict(self, texts: list[str]) -> list[Prediction]:
        ensure_non_empty(texts)
        probabilities = self._pipeline.predict_proba(texts)
        return build_predictions(self._reorder(probabilities))

    def _reorder(self, probabilities: np.ndarray) -> np.ndarray:
        """Garante que as colunas sigam a ordem de LABELS, não a de classes_."""
        classes = list(self._pipeline.named_steps["clf"].classes_)
        order = [classes.index(label) for label in LABELS]
        return probabilities[:, order]
