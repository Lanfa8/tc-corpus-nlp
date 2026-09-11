import pytest
from fastapi.testclient import TestClient

EXPECTED_METRICS = [
    "triage_requests_total",
    "triage_request_duration_seconds",
    "triage_inference_duration_seconds",
    "triage_predictions_total",
    "triage_errors_total",
]


def test_metrics_endpoint_is_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.parametrize("metric", EXPECTED_METRICS)
def test_expected_metric_families_are_declared(client, metric):
    client.post("/predict", json={"text": "tumor malignant growth"})
    body = client.get("/metrics").text
    assert metric in body


def test_request_counter_increments(client):
    client.post("/predict", json={"text": "cardiac heart arterial coronary"})
    body = client.get("/metrics").text
    assert 'triage_requests_total{endpoint="/predict",status="200"}' in body


def test_prediction_counter_carries_condition_and_urgency(client):
    client.post("/predict", json={"text": "tumor malignant carcinoma metastasis"})
    body = client.get("/metrics").text
    assert 'condition="neoplasms"' in body
    assert 'urgency="urgente"' in body


def test_inference_duration_is_labelled_by_backend(client):
    client.post("/predict", json={"text": "gastric intestinal bowel ulcer"})
    body = client.get("/metrics").text
    assert 'triage_inference_duration_seconds_count{backend="sklearn"}' in body


def test_error_counter_increments_on_degraded_predict(degraded_client):
    degraded_client.post("/predict", json={"text": "qualquer laudo"})
    body = degraded_client.get("/metrics").text
    assert 'triage_errors_total{endpoint="/predict"' in body


def test_metrics_endpoint_is_not_self_counted(client):
    client.get("/metrics")
    body = client.get("/metrics").text
    assert 'endpoint="/metrics"' not in body


def test_unhandled_exception_is_recorded_and_reraised(client, monkeypatch):
    def boom(_texts):
        raise ValueError("erro inesperado de inferência")

    monkeypatch.setattr(client.app.state.predictor, "predict", boom)
    raw_client = TestClient(client.app, raise_server_exceptions=False)

    response = raw_client.post("/predict", json={"text": "qualquer laudo"})
    assert response.status_code == 500

    body = raw_client.get("/metrics").text
    assert 'triage_errors_total{endpoint="/predict",type="unhandled_exception"}' in body
    assert 'triage_requests_total{endpoint="/predict",status="500"}' in body
