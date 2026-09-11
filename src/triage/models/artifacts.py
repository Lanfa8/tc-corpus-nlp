"""Persistência do pipeline treinado e dos seus metadados."""

import json
import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PIPELINE_FILE = "pipeline.joblib"
METADATA_FILE = "metadata.json"


def save_pipeline(pipeline: Pipeline, metadata: dict, models_dir: Path) -> None:
    """Salva o pipeline e os metadados no diretório de modelos."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / PIPELINE_FILE)
    (models_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("artefatos salvos em %s", models_dir)


def load_pipeline(models_dir: Path) -> tuple[Pipeline, dict]:
    """Carrega o pipeline e os metadados; levanta FileNotFoundError se faltarem."""
    models_dir = Path(models_dir)
    pipeline_path = models_dir / PIPELINE_FILE
    metadata_path = models_dir / METADATA_FILE
    if not pipeline_path.exists():
        raise FileNotFoundError(f"pipeline não encontrado em {pipeline_path}")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return joblib.load(pipeline_path), metadata
