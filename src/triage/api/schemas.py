"""Contratos de entrada e saída da API."""

from pydantic import BaseModel, Field

from triage.inference.base import Prediction


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto do laudo médico")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    condition_label: int
    condition_name: str
    urgency: str
    confidence: float
    probabilities: dict[str, float]

    @classmethod
    def from_prediction(cls, prediction: Prediction) -> "PredictionResponse":
        return cls(
            condition_label=prediction.condition_label,
            condition_name=prediction.condition_name,
            urgency=prediction.urgency,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
        )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str


class ModelInfoResponse(BaseModel):
    backend: str
    model_name: str | None = None
    trained_at: str | None = None
    macro_f1: float | None = None
