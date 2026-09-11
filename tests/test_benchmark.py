from scripts.benchmark_latency import benchmark
from triage.inference.predictor import SklearnPredictor
from triage.models.artifacts import save_pipeline
from triage.models.factory import build_pipeline


def test_benchmark_returns_expected_keys(tmp_path, synthetic_frame):
    pipeline = build_pipeline("logistic_regression", min_df=1)
    pipeline.fit(
        synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
    )
    save_pipeline(pipeline, {}, tmp_path)
    predictor = SklearnPredictor.from_artifacts(tmp_path)

    result = benchmark(predictor, ["tumor malignant growth"], n_warmup=2, n_runs=10)

    assert set(result) == {
        "backend",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "throughput_rps",
        "n_runs",
    }
    assert result["backend"] == "sklearn"
    assert result["n_runs"] == 10
    assert result["p50_ms"] > 0
    assert result["p99_ms"] >= result["p50_ms"]
    assert result["throughput_rps"] > 0
