import pytest

pytest.importorskip("airflow.models", reason="Airflow roda apenas no container")


def test_dag_imports_without_errors():
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder="airflow/dags", include_examples=False)
    assert dagbag.import_errors == {}
    assert "triage_training" in dagbag.dags


def test_dag_task_order():
    from airflow.models import DagBag

    dag = DagBag(dag_folder="airflow/dags", include_examples=False).dags[
        "triage_training"
    ]
    assert set(dag.task_ids) == {
        "validate_data",
        "load_data",
        "train_model",
        "evaluate_model",
        "export_onnx",
        "register_artifacts",
    }
    assert dag.get_task("train_model").upstream_task_ids == {"load_data"}
    assert dag.get_task("export_onnx").upstream_task_ids == {"evaluate_model"}
