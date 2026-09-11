"""Resolução do backend de inferência."""

from pathlib import Path

from triage.inference.base import Prediction, Predictor

__all__ = ["Prediction", "Predictor", "load_predictor"]


def load_predictor(backend: str, models_dir: Path) -> Predictor:
    """Instancia o backend pedido. Importa sob demanda para não exigir onnxruntime
    quando o backend sklearn está em uso, e vice-versa."""
    if backend == "sklearn":
        from triage.inference.predictor import SklearnPredictor

        return SklearnPredictor.from_artifacts(models_dir)
    if backend == "onnx":
        from triage.inference.onnx_predictor import OnnxPredictor

        return OnnxPredictor.from_artifacts(models_dir)
    raise ValueError(f"backend desconhecido: {backend!r}; use 'sklearn' ou 'onnx'")
