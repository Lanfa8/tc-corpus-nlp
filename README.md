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

O stack roda os contêineres do Airflow com o UID do host (`AIRFLOW_UID`),
não com o usuário embutido na imagem oficial — assim `models/` continua
pertencendo ao seu usuário e permanece gravável tanto pelo host (`make
train`) quanto pelos contêineres, sem precisar de `chown` na árvore
inteira. `make airflow-up` grava `AIRFLOW_UID=$(id -u)` em `.env`
automaticamente na primeira execução, se a variável ainda não estiver lá.
`airflow-init` roda como `root` só para criar e ajustar o dono de
`airflow/logs/` e de `models/staging/` (criado pela própria DAG), nunca da
árvore `models/` inteira.
