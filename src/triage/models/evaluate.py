"""Avaliação do classificador de triagem."""

import logging

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from triage.data.schema import CONDITION_NAMES, LABELS

logger = logging.getLogger(__name__)


def evaluate(pipeline: Pipeline, texts: list[str], labels: list[int]) -> dict:
    """Calcula macro-F1, acurácia, métricas por classe e matriz de confusão."""
    predictions = pipeline.predict(texts)
    target_names = [CONDITION_NAMES[label] for label in LABELS]

    report = classification_report(
        labels,
        predictions,
        labels=LABELS,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=LABELS
        ).tolist(),
    }
    logger.info("macro_f1=%.4f accuracy=%.4f", result["macro_f1"], result["accuracy"])
    return result
