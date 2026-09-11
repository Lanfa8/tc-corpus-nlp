"""Treino do classificador e comparação entre candidatos."""

import logging
import time

import pandas as pd
from sklearn.pipeline import Pipeline

from triage.models.evaluate import evaluate
from triage.models.factory import build_pipeline

logger = logging.getLogger(__name__)


def train(
    texts: list[str], labels: list[int], model_name: str, **vectorizer_kwargs
) -> Pipeline:
    """Treina um pipeline no conjunto informado."""
    pipeline = build_pipeline(model_name, **vectorizer_kwargs)
    logger.info("treinando %s em %d exemplos", model_name, len(texts))
    pipeline.fit(texts, labels)
    return pipeline


def run_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_names: list[str],
    **vectorizer_kwargs,
) -> list[dict]:
    """Treina cada candidato e retorna os resultados ordenados por macro-F1."""
    results = []
    for model_name in model_names:
        started = time.perf_counter()
        pipeline = train(
            train_df["medical_abstract"].tolist(),
            train_df["condition_label"].tolist(),
            model_name,
            **vectorizer_kwargs,
        )
        fit_seconds = time.perf_counter() - started
        report = evaluate(
            pipeline,
            val_df["medical_abstract"].tolist(),
            val_df["condition_label"].tolist(),
        )
        results.append(
            {
                "model_name": model_name,
                "macro_f1": report["macro_f1"],
                "accuracy": report["accuracy"],
                "fit_seconds": fit_seconds,
            }
        )
    return sorted(results, key=lambda r: r["macro_f1"], reverse=True)
