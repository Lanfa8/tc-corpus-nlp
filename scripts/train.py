"""Treina o classificador de triagem e salva os artefatos.

Uso: python scripts/train.py [--model logistic_regression]
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from triage.config.seeds import SEED, set_seed
from triage.config.settings import get_settings
from triage.data.loader import load_split, train_val_split
from triage.models.artifacts import save_pipeline
from triage.models.evaluate import evaluate
from triage.models.train import run_experiments, train

logger = logging.getLogger(__name__)


def load_params(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o classificador de triagem")
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="compara candidatos")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    set_seed()

    params = load_params(args.params)
    vectorizer_kwargs = {
        "ngram_range": tuple(params["features"]["ngram_range"]),
        "min_df": params["features"]["min_df"],
        "max_df": params["features"]["max_df"],
        "max_features": params["features"]["max_features"],
    }

    full_train = load_split("train", data_dir=settings.data_dir)
    train_df, val_df = train_val_split(
        full_train, val_size=params["data"]["val_size"], seed=SEED
    )

    model_name = args.model or params["model"]["name"]
    comparison = []
    if args.compare:
        comparison = run_experiments(
            train_df,
            val_df,
            model_names=["dummy", "naive_bayes", "logistic_regression", "linear_svc"],
            **vectorizer_kwargs,
        )
        logger.info("comparação:\n%s", json.dumps(comparison, indent=2))
        model_name = comparison[0]["model_name"]

    # Modelo final: treinado no conjunto completo de treino.
    pipeline = train(
        full_train["medical_abstract"].tolist(),
        full_train["condition_label"].tolist(),
        model_name,
        **vectorizer_kwargs,
    )

    test_df = load_split("test", data_dir=settings.data_dir)
    holdout = evaluate(
        pipeline,
        test_df["medical_abstract"].tolist(),
        test_df["condition_label"].tolist(),
    )

    minimum = params["evaluation"]["min_macro_f1"]
    if holdout["macro_f1"] < minimum:
        raise SystemExit(
            f"macro_f1={holdout['macro_f1']:.4f} abaixo do mínimo {minimum}"
        )

    save_pipeline(
        pipeline,
        {
            "model_name": model_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "n_train": len(full_train),
            "vectorizer": {
                **vectorizer_kwargs,
                "ngram_range": list(vectorizer_kwargs["ngram_range"]),
            },
            "holdout": holdout,
            "comparison": comparison,
        },
        settings.models_dir,
    )


if __name__ == "__main__":
    main()
