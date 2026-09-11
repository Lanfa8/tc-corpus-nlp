# triage

Triagem automática de laudos médicos com NLP, servida via FastAPI.

## Problema

Classificação automática de resumos de laudos médicos (abstracts) em uma
de cinco categorias clínicas, com o objetivo de apoiar a priorização e o
roteamento inicial de casos em um fluxo de triagem.

## Dataset

Baseado no **Medical Abstracts TC Corpus**, com 5 classes:

1. neoplasms
2. digestive system diseases
3. nervous system diseases
4. cardiovascular diseases
5. general pathological conditions

Os arquivos CSV (`data/medical_tc_train.csv`, `data/medical_tc_test.csv`,
`data/medical_tc_labels.csv`) não são versionados no repositório — devem
ser obtidos separadamente e colocados em `data/`.

## Pré-requisitos

- Python 3.10
- [Poetry](https://python-poetry.org/) 2.x
- Docker e Docker Compose (para execução em contêiner)

## Como executar

```bash
# instalar dependências
make install

# treinar o modelo (gera models/pipeline.joblib e models/metadata.json)
make train

# construir a imagem Docker do serviço de inferência
make docker-build

# rodar o serviço via Docker (monta models/ como volume)
make docker-run
```

O serviço fica disponível em `http://localhost:8000`, com `GET /health`
para verificação de status.

## Orquestração com Airflow

O treino/retreino é orquestrado por uma DAG do Airflow (`triage_training`),
em um stack `docker-compose.airflow.yml` separado do serviço de inferência.

```bash
mkdir -p airflow/logs
make airflow-up
make airflow-down
```

O stack usa `AIRFLOW_UID` (default `50000`, o usuário embutido na imagem
oficial do Airflow) para rodar os contêineres e ajustar a dono dos
diretórios montados (`airflow/logs/`, `models/`) via `airflow-init`, que
roda como `root` e faz `chown` neles antes de subir o scheduler/webserver —
não é necessário nenhum ajuste manual de permissão no host. Se o seu UID
local difere e você quiser que os arquivos fiquem com o seu usuário fora
dos contêineres, rode `echo "AIRFLOW_UID=$(id -u)" >> .env` antes de
`make airflow-up`.
