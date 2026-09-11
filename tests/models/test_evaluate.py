from triage.models.evaluate import evaluate
from triage.models.factory import build_pipeline


def test_evaluate_returns_expected_keys(synthetic_frame):
    pipeline = build_pipeline("logistic_regression", min_df=1)
    texts = synthetic_frame["medical_abstract"].tolist()
    labels = synthetic_frame["condition_label"].tolist()
    pipeline.fit(texts, labels)

    report = evaluate(pipeline, texts, labels)

    assert set(report) == {"macro_f1", "accuracy", "per_class", "confusion_matrix"}
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert len(report["confusion_matrix"]) == 5
    assert set(report["per_class"]) == {
        "neoplasms",
        "digestive system diseases",
        "nervous system diseases",
        "cardiovascular diseases",
        "general pathological conditions",
    }


def test_evaluate_learns_the_synthetic_corpus(synthetic_frame):
    """Sanidade: o corpus sintético é separável, então o macro-F1 deve ser alto."""
    pipeline = build_pipeline("logistic_regression", min_df=1)
    texts = synthetic_frame["medical_abstract"].tolist()
    labels = synthetic_frame["condition_label"].tolist()
    pipeline.fit(texts, labels)
    assert evaluate(pipeline, texts, labels)["macro_f1"] > 0.9


def test_evaluate_is_json_serializable(synthetic_frame):
    import json

    pipeline = build_pipeline("naive_bayes", min_df=1)
    texts = synthetic_frame["medical_abstract"].tolist()
    labels = synthetic_frame["condition_label"].tolist()
    pipeline.fit(texts, labels)
    json.dumps(evaluate(pipeline, texts, labels))
