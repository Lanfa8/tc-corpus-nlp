from sklearn.pipeline import Pipeline

from triage.data.loader import train_val_split
from triage.models.train import run_experiments, train


def test_train_returns_fitted_pipeline(synthetic_frame):
    pipeline = train(
        synthetic_frame["medical_abstract"].tolist(),
        synthetic_frame["condition_label"].tolist(),
        model_name="logistic_regression",
        min_df=1,
    )
    assert isinstance(pipeline, Pipeline)
    assert pipeline.predict(["cardiac heart arterial"])[0] in range(1, 6)


def test_run_experiments_ranks_by_macro_f1(synthetic_frame):
    train_df, val_df = train_val_split(synthetic_frame, val_size=0.25)
    results = run_experiments(
        train_df, val_df, model_names=["dummy", "logistic_regression"], min_df=1
    )

    assert [r["model_name"] for r in results] == ["logistic_regression", "dummy"]
    assert results[0]["macro_f1"] > results[1]["macro_f1"]
    assert set(results[0]) == {"model_name", "macro_f1", "accuracy", "fit_seconds"}
