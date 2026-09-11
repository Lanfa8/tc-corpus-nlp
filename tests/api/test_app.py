from triage.urgency import URGENCY_LEVELS


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "sklearn"


def test_predict_returns_condition_and_urgency(client):
    response = client.post(
        "/predict", json={"text": "tumor malignant carcinoma metastasis oncology"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["condition_name"] == "neoplasms"
    assert body["urgency"] in URGENCY_LEVELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 5


def test_predict_sets_process_time_header(client):
    response = client.post("/predict", json={"text": "cardiac heart arterial"})
    assert "x-process-time-ms" in response.headers
    assert float(response.headers["x-process-time-ms"]) >= 0


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422


def test_batch_predict_returns_one_per_text(client):
    response = client.post(
        "/predict/batch",
        json={"texts": ["tumor malignant growth", "gastric intestinal bowel ulcer"]},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_batch_predict_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_model_info_exposes_metadata(client):
    body = client.get("/model/info").json()
    assert body["backend"] == "sklearn"
    assert body["model_name"] == "logistic_regression"
    assert body["macro_f1"] == 0.99


def test_model_info_falls_back_to_validation_metadata(client_with_validation_metadata):
    body = client_with_validation_metadata.get("/model/info").json()
    assert body["macro_f1"] == 0.68


def test_degraded_mode_starts_without_artifacts(degraded_client):
    body = degraded_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_degraded_mode_returns_503_on_predict(degraded_client):
    response = degraded_client.post("/predict", json={"text": "qualquer laudo"})
    assert response.status_code == 503
