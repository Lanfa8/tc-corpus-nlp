# Relatório de latência: sklearn vs. ONNX Runtime

Comparação de latência entre o pipeline scikit-learn original
(`pipeline.joblib`) e sua exportação para ONNX (`pipeline.onnx`).

## 1. Metodologia

- `make benchmark` executa `scripts/benchmark_latency.py` e grava os
  números brutos em `docs/latency_results.json`.
- Textos reais do held-out (2.770 documentos), 20 execuções de aquecimento
  descartadas e 300 medições por backend (60 no caso batch).
- Os dois backends rodam com uma thread, para medir o custo por requisição
  e não o paralelismo de cada biblioteca.
- Máquina: Intel Core i7-1355U, WSL2, execução local via Poetry.

## 2. Baseline HTTP

Antes da otimização, a latência **fim a fim** foi medida com 50 requisições
contra `POST /predict` rodando em Docker:

| Métrica | Valor |
|---|---|
| p50 | 6.4 ms |
| p95 | 8.4 ms |
| p99 | 8.9 ms |

Esse número inclui rede, parsing HTTP e serialização JSON, não só o modelo.
Por isso a comparação abaixo mede apenas `predictor.predict()`.

## 3. Resultados

### Requisição única (1 documento)

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | throughput (req/s) | artefato |
|---|---|---|---|---|---|
| sklearn | 0.80 | 1.09 | 1.45 | 1 202 | 6.0 MB |
| onnx | 0.26 | 0.34 | 0.45 | 3 726 | 2.7 MB |
| **speedup** | **3.13x** | 3.21x | 3.20x | 3.10x | — |

### Batch (32 documentos)

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | throughput (req/s) |
|---|---|---|---|---|
| sklearn | 6.83 | 8.55 | 9.36 | 4 534 |
| onnx | 12.85 | 15.11 | 27.14 | 2 462 |
| **speedup** | **0.53x (ONNX mais lento)** | 0.57x | 0.34x | 0.54x |

## 4. Paridade

Sobre o held-out completo, os dois backends concordam em **100%** das
classes, com diferença máxima de probabilidade de **2.813e-07**. A
otimização não muda o comportamento do modelo.

## 5. Análise e decisão

- **Requisição única**: o ONNX executa vetorização e classificação num
  único grafo em C++, sem voltar ao Python entre as etapas. Para um
  documento, esse overhead fixo domina, daí o ganho de ~3x.
- **Batch**: o sklearn ganha. O `scipy.sparse` é muito eficiente na
  multiplicação esparsa do TF-IDF, e o grafo ONNX exportado não preserva
  essa representação em todo o caminho.
- **Ganho percebido pelo cliente**: os ~0.55 ms economizados são pequenos
  frente aos ~6.4 ms do p50 fim a fim, porque o overhead HTTP não muda com
  o backend.

**Backend padrão: `onnx`.** O caso de uso central é `/predict` (uma triagem
por vez), onde o ONNX é mais rápido. Para uso intenso de `/predict/batch`,
`TRIAGE_BACKEND=sklearn` é a melhor escolha.
