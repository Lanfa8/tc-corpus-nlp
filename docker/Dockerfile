# ---- builder ----
FROM python:3.10-slim AS builder

ENV POETRY_VERSION=2.4.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# ---- runtime ----
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    TRIAGE_MODELS_DIR=/app/models

WORKDIR /app

RUN useradd --create-home --uid 1000 triage

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs

RUN mkdir -p /app/models /app/data && chown -R triage:triage /app

USER triage
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
