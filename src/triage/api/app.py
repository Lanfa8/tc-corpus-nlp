"""Serviço de triagem de laudos médicos."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from triage.api import metrics
from triage.api.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictRequest,
)
from triage.config.settings import get_settings
from triage.inference import Predictor, load_predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo no boot; degrada em vez de derrubar o serviço."""
    settings = get_settings()
    app.state.backend = settings.backend
    try:
        app.state.predictor = load_predictor(settings.backend, settings.models_dir)
        logger.info("modelo carregado (backend=%s)", settings.backend)
    except (FileNotFoundError, OSError) as error:
        app.state.predictor = None
        logger.warning("subindo em modo degradado: %s", error)
    yield
    app.state.predictor = None


def get_predictor(request: Request) -> Predictor:
    """Dependência que exige um modelo carregado."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="modelo indisponível: artefatos não carregados"
        )
    return predictor


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    app = FastAPI(
        title="Triagem de Laudos Médicos",
        description=(
            "Classifica laudos em 5 condições clínicas e deriva um nível de "
            "urgência de triagem. Heurística de apoio — não substitui avaliação "
            "de profissional de saúde."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        return response

    metrics.install(app)

    @app.get("/health", response_model=HealthResponse, tags=["infra"])
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.predictor is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            backend=request.app.state.backend,
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["infra"])
    def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
        metadata = predictor.metadata
        return ModelInfoResponse(
            backend=predictor.backend,
            model_name=metadata.get("model_name"),
            trained_at=metadata.get("trained_at"),
            macro_f1=(metadata.get("holdout") or metadata.get("validation") or {}).get(
                "macro_f1"
            ),
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["triagem"])
    def predict(
        payload: PredictRequest, predictor: Predictor = Depends(get_predictor)
    ) -> PredictionResponse:
        with metrics.observe_inference(predictor.backend):
            predictions = predictor.predict([payload.text])
        metrics.record_predictions(predictions)
        return PredictionResponse.from_prediction(predictions[0])

    @app.post(
        "/predict/batch", response_model=BatchPredictionResponse, tags=["triagem"]
    )
    def predict_batch(
        payload: BatchPredictRequest, predictor: Predictor = Depends(get_predictor)
    ) -> BatchPredictionResponse:
        with metrics.observe_inference(predictor.backend):
            predictions = predictor.predict(payload.texts)
        metrics.record_predictions(predictions)
        return BatchPredictionResponse(
            predictions=[PredictionResponse.from_prediction(p) for p in predictions]
        )

    return app


app = create_app()
