"""Compara a latência de inferência entre os backends sklearn e ONNX.

Também reproduz, de forma automatizada, a medição de paridade entre os dois
backends (concordância de classe e diferença máxima de probabilidade) sobre o
conjunto de teste (held-out) completo — não apenas uma amostra — para que a
alegação central da Etapa 4 seja verificável a partir de código versionado,
não apenas de prosa em um relatório.

Uso: python scripts/benchmark_latency.py [--runs 300] [--batch-size 32]
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from triage.config.seeds import set_seed
from triage.config.settings import get_settings
from triage.data.loader import load_split
from triage.inference import load_predictor
from triage.inference.base import Predictor

logger = logging.getLogger(__name__)


def benchmark(
    predictor: Predictor, texts: list[str], n_warmup: int = 20, n_runs: int = 300
) -> dict:
    """Mede a latência de `predictor.predict(texts)` repetidas vezes."""
    for _ in range(n_warmup):
        predictor.predict(texts)

    durations_ms = []
    for _ in range(n_runs):
        started = time.perf_counter()
        predictor.predict(texts)
        durations_ms.append((time.perf_counter() - started) * 1000)

    durations_ms.sort()
    mean_ms = statistics.fmean(durations_ms)
    return {
        "backend": predictor.backend,
        "p50_ms": durations_ms[int(n_runs * 0.50)],
        "p95_ms": durations_ms[int(n_runs * 0.95)],
        "p99_ms": durations_ms[min(int(n_runs * 0.99), n_runs - 1)],
        "mean_ms": mean_ms,
        "throughput_rps": len(texts) / (mean_ms / 1000),
        "n_runs": n_runs,
    }


def compare_parity(
    sklearn_predictor: Predictor, onnx_predictor: Predictor, texts: list[str]
) -> dict:
    """Compara predições dos dois backends sobre o conjunto completo `texts`.

    Retorna a fração de concordância de classe e a diferença absoluta máxima
    de probabilidade — as duas métricas por trás da alegação de paridade da
    Etapa 4, medidas aqui em código, não apenas relatadas em prosa.
    """
    sklearn_predictions = sklearn_predictor.predict(texts)
    onnx_predictions = onnx_predictor.predict(texts)

    agreement = sum(
        a.condition_label == b.condition_label
        for a, b in zip(sklearn_predictions, onnx_predictions, strict=True)
    ) / len(texts)

    max_abs_prob_diff = max(
        abs(a.probabilities[name] - b.probabilities[name])
        for a, b in zip(sklearn_predictions, onnx_predictions, strict=True)
        for name in a.probabilities
    )

    return {
        "agreement": agreement,
        "max_abs_prob_diff": float(max_abs_prob_diff),
        "n_docs": len(texts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de latência dos backends")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/latency_results.json")
    )
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    set_seed()
    settings = get_settings()

    sample = load_split("test", data_dir=settings.data_dir)["medical_abstract"].tolist()
    single = sample[:1]
    batch = sample[: args.batch_size]

    results = {"single": [], "batch": [], "artifact_bytes": {}}
    predictors = {}
    for backend in ("sklearn", "onnx"):
        predictor = load_predictor(backend, settings.models_dir)
        predictors[backend] = predictor
        results["single"].append(benchmark(predictor, single, n_runs=args.runs))
        results["batch"].append(
            benchmark(predictor, batch, n_runs=max(args.runs // 5, 20))
        )

    for backend, filename in (
        ("sklearn", "pipeline.joblib"),
        ("onnx", "pipeline.onnx"),
    ):
        path = settings.models_dir / filename
        results["artifact_bytes"][backend] = path.stat().st_size if path.exists() else 0

    # Paridade entre backends sobre o conjunto de teste COMPLETO — a
    # otimização não pode mudar o modelo. (amendment A: full set, not [:500])
    sklearn_predictor = predictors["sklearn"]
    onnx_predictor = predictors["onnx"]
    parity = compare_parity(sklearn_predictor, onnx_predictor, sample)
    results["parity"] = parity
    # Mantido por compatibilidade com consumidores que leem `agreement` no
    # nível superior do JSON.
    results["agreement"] = parity["agreement"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for mode in ("single", "batch"):
        base, optimized = results[mode]
        logger.info(
            "%s: %s p50=%.3fms -> %s p50=%.3fms (%.2fx)",
            mode,
            base["backend"],
            base["p50_ms"],
            optimized["backend"],
            optimized["p50_ms"],
            base["p50_ms"] / optimized["p50_ms"],
        )
    logger.info(
        "paridade sobre %d docs do held-out: concordância=%.4f "
        "diferença máxima de probabilidade=%.3e",
        parity["n_docs"],
        parity["agreement"],
        parity["max_abs_prob_diff"],
    )


if __name__ == "__main__":
    main()
