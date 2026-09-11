from pathlib import Path

from triage.config.settings import Settings, get_settings


def test_defaults():
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.models_dir == Path("models")
    assert settings.backend == "onnx"


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "onnx")
    monkeypatch.setenv("TRIAGE_MODELS_DIR", "/tmp/models")
    settings = Settings()
    assert settings.backend == "onnx"
    assert settings.models_dir == Path("/tmp/models")


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
