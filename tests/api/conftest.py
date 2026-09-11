import pytest
from fastapi.testclient import TestClient

from triage.models.artifacts import save_pipeline
from triage.models.factory import build_pipeline


@pytest.fixture
def trained_models_dir(tmp_path, synthetic_frame):
    pipeline = build_pipeline("logistic_regression", min_df=1)
    pipeline.fit(
        synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
    )
    save_pipeline(
        pipeline,
        {
            "model_name": "logistic_regression",
            "trained_at": "2026-09-10T00:00:00+00:00",
            "holdout": {"macro_f1": 0.99},
        },
        tmp_path,
    )
    return tmp_path


@pytest.fixture
def client_with_validation_metadata(tmp_path, synthetic_frame, monkeypatch):
    """Cliente cujo metadata guarda métricas sob `validation` (formato DAG)."""
    pipeline = build_pipeline("logistic_regression", min_df=1)
    pipeline.fit(
        synthetic_frame["medical_abstract"], synthetic_frame["condition_label"]
    )
    save_pipeline(
        pipeline,
        {
            "model_name": "logistic_regression",
            "trained_at": "2026-09-10T00:00:00+00:00",
            "validation": {"macro_f1": 0.68},
        },
        tmp_path,
    )
    monkeypatch.setenv("TRIAGE_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGE_BACKEND", "sklearn")
    from triage.config.settings import get_settings

    get_settings.cache_clear()
    from triage.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def client(trained_models_dir, monkeypatch):
    """Cliente com artefatos disponíveis."""
    monkeypatch.setenv("TRIAGE_MODELS_DIR", str(trained_models_dir))
    monkeypatch.setenv("TRIAGE_BACKEND", "sklearn")
    from triage.config.settings import get_settings

    get_settings.cache_clear()
    from triage.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def degraded_client(tmp_path, monkeypatch):
    """Cliente sem artefatos — a API deve subir em modo degradado."""
    monkeypatch.setenv("TRIAGE_MODELS_DIR", str(tmp_path / "vazio"))
    from triage.config.settings import get_settings

    get_settings.cache_clear()
    from triage.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
