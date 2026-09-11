FROM apache/airflow:2.9.3-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY --chown=airflow:root requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

ENV PYTHONPATH=/opt/project/src
