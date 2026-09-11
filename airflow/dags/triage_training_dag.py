"""DAG de treino/retreino do classificador de triagem.

validate_data -> load_data -> train_model -> evaluate_model -> export_onnx
              -> register_artifacts

`evaluate_model` funciona como gate: se o macro-F1 na validação ficar abaixo
do mínimo configurado, a DAG falha e o artefato em produção não é substituído.
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/project")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STAGING_DIR = PROJECT_ROOT / "models" / "staging"
PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"

DEFAULT_ARGS = {
    "owner": "mlet",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))


@dag(
    dag_id="triage_training",
    description="Ingestão, treino, avaliação e exportação do modelo de triagem",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    default_args=DEFAULT_ARGS,
    tags=["mlet", "nlp", "triagem"],
)
def triage_training():
    @task
    def validate_data() -> dict:
        """Confere que os CSVs do corpus existem e não estão vazios."""
        params = _params()
        sizes = {}
        for key in ("train_file", "test_file"):
            path = DATA_DIR / params["data"][key]
            if not path.exists():
                raise FileNotFoundError(f"arquivo de dados ausente: {path}")
            size = path.stat().st_size
            if size == 0:
                raise ValueError(f"arquivo de dados vazio: {path}")
            sizes[key] = size
        logger.info("dados validados: %s", sizes)
        return sizes

    @task
    def load_data() -> dict:
        """Carrega e limpa o corpus, materializando train/val em parquet."""
        from triage.data.loader import load_split, train_val_split

        params = _params()
        full_train = load_split("train", data_dir=DATA_DIR)
        train_df, val_df = train_val_split(
            full_train, val_size=params["data"]["val_size"]
        )

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        train_path = STAGING_DIR / "train.parquet"
        val_path = STAGING_DIR / "val.parquet"
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)

        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "n_train": len(train_df),
            "n_val": len(val_df),
        }

    @task
    def train_model(data: dict) -> str:
        """Treina o pipeline e salva em staging."""
        import pandas as pd

        from triage.config.seeds import set_seed
        from triage.models.artifacts import save_pipeline
        from triage.models.train import train

        set_seed()
        params = _params()
        train_df = pd.read_parquet(data["train_path"])

        pipeline = train(
            train_df["medical_abstract"].tolist(),
            train_df["condition_label"].tolist(),
            model_name=params["model"]["name"],
            ngram_range=tuple(params["features"]["ngram_range"]),
            min_df=params["features"]["min_df"],
            max_df=params["features"]["max_df"],
            max_features=params["features"]["max_features"],
        )
        save_pipeline(
            pipeline,
            {
                "model_name": params["model"]["name"],
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_train": data["n_train"],
            },
            STAGING_DIR,
        )
        return str(STAGING_DIR)

    @task
    def evaluate_model(staging_dir: str, data: dict) -> dict:
        """Gate de qualidade: falha se o macro-F1 ficar abaixo do mínimo."""
        import pandas as pd

        from triage.models.artifacts import load_pipeline, save_pipeline
        from triage.models.evaluate import evaluate

        params = _params()
        pipeline, metadata = load_pipeline(Path(staging_dir))
        val_df = pd.read_parquet(data["val_path"])

        report = evaluate(
            pipeline,
            val_df["medical_abstract"].tolist(),
            val_df["condition_label"].tolist(),
        )
        minimum = params["evaluation"]["min_macro_f1"]
        if report["macro_f1"] < minimum:
            raise ValueError(
                f"macro_f1={report['macro_f1']:.4f} abaixo do mínimo {minimum}; "
                "modelo não promovido"
            )

        metadata["holdout"] = report
        save_pipeline(pipeline, metadata, Path(staging_dir))
        logger.info("gate aprovado: macro_f1=%.4f", report["macro_f1"])
        return report

    @task
    def export_onnx() -> str:
        """Exporta o pipeline de staging para ONNX."""
        result = subprocess.run(
            [
                "python",
                str(PROJECT_ROOT / "scripts" / "export_onnx.py"),
                "--models-dir",
                str(STAGING_DIR),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        logger.info(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"export para ONNX falhou:\n{result.stderr}")
        return str(STAGING_DIR / "pipeline.onnx")

    @task
    def register_artifacts(staging_dir: str, report: dict) -> dict:
        """Promove os artefatos de staging para o diretório servido pela API."""
        staging = Path(staging_dir)
        for filename in ("pipeline.joblib", "pipeline.onnx", "metadata.json"):
            source = staging / filename
            if source.exists():
                shutil.copy2(source, MODELS_DIR / filename)
                logger.info("promovido: %s", filename)

        summary = {
            "macro_f1": report["macro_f1"],
            "accuracy": report["accuracy"],
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        (MODELS_DIR / "last_run.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    validated = validate_data()
    data = load_data()
    validated >> data

    staging = train_model(data)
    report = evaluate_model(staging, data)
    onnx = export_onnx()
    promoted = register_artifacts(staging, report)
    report >> onnx
    onnx >> promoted


triage_training()
