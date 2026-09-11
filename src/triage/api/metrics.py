"""Instrumentação Prometheus do serviço de triagem."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from triage.inference.base import Prediction

logger = logging.getLogger(__name__)

# Buckets ajustados à escala real do serviço (1ms a 1s). Os defaults do cliente
# começam em 5ms e perderiam resolução justamente na faixa onde o ganho do ONNX
# aparece.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

REQUESTS_TOTAL = Counter(
    "triage_requests_total",
    "Total de requisições HTTP atendidas",
    ["endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "triage_request_duration_seconds",
    "Duração da requisição HTTP fim a fim",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)

INFERENCE_DURATION = Histogram(
    "triage_inference_duration_seconds",
    "Duração apenas da inferência do modelo",
    ["backend"],
    buckets=LATENCY_BUCKETS,
)

PREDICTIONS_TOTAL = Counter(
    "triage_predictions_total",
    "Total de laudos classificados por condição e urgência",
    ["condition", "urgency"],
)

ERRORS_TOTAL = Counter(
    "triage_errors_total",
    "Total de respostas de erro",
    ["endpoint", "type"],
)

EXCLUDED_PATHS = {"/metrics", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


@contextmanager
def observe_inference(backend: str) -> Iterator[None]:
    """Cronometra o trecho de inferência do modelo."""
    started = time.perf_counter()
    try:
        yield
    finally:
        INFERENCE_DURATION.labels(backend=backend).observe(
            time.perf_counter() - started
        )


def record_predictions(predictions: list[Prediction]) -> None:
    """Contabiliza as classes e urgências preditas."""
    for prediction in predictions:
        PREDICTIONS_TOTAL.labels(
            condition=prediction.condition_name, urgency=prediction.urgency
        ).inc()


def _route_template(request: Request) -> str:
    """Usa o template da rota (não a URL concreta) para não explodir a cardinalidade."""
    route = request.scope.get("route")
    if route is None:
        return "__not_found__"
    return getattr(route, "path", "__not_found__")


def install(app: FastAPI) -> None:
    """Instala o middleware de métricas e a rota de exposição."""

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            endpoint = _route_template(request)
            logger.exception(
                "erro não tratado ao processar %s %s", request.method, endpoint
            )
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="500").inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(elapsed)
            ERRORS_TOTAL.labels(endpoint=endpoint, type="unhandled_exception").inc()
            raise

        elapsed = time.perf_counter() - started
        endpoint = _route_template(request)
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=response.status_code).inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(elapsed)
        if response.status_code >= 400:
            ERRORS_TOTAL.labels(
                endpoint=endpoint, type=f"http_{response.status_code}"
            ).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
