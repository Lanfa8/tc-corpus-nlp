# Relatório de latência: sklearn vs. ONNX Runtime

Este relatório documenta a comparação de latência entre os dois backends de
inferência do serviço de triagem — o pipeline scikit-learn original
(`pipeline.joblib`) e sua exportação para um único grafo ONNX
(`pipeline.onnx`) — e reproduz, a partir de código versionado, a alegação de
paridade entre eles.

`make benchmark` executa `scripts/benchmark_latency.py` e grava os números
brutos em `docs/latency_results.json`; os números deste relatório vêm
diretamente desse arquivo, gerado localmente com o comando abaixo. Não há
número aqui estimado ou re-derivado fora do próprio script.

## 1. Metodologia

- **Hardware**: CPU Intel Core i7-1355U (13ª geração), 12 threads lógicos
  disponíveis ao processo, 7.6 GiB de RAM.
- **Sistema operacional**: Linux (WSL2, kernel
  `6.18.33.2-microsoft-standard-WSL2`), execução local via Poetry (não em
  contêiner Docker — ver seção 2 para o baseline que *foi* medido em Docker).
- **Versões**: `scikit-learn==1.7.2`, `onnxruntime==1.23.2`,
  `skl2onnx==1.20.0`, `numpy==1.26.4`.
- **Textos**: amostras reais do held-out (`data/medical_tc_test.csv`), carregadas
  via `triage.data.loader.load_split("test")`. Após a limpeza (remoção de
  duplicados), o split de teste tem **2.770 documentos**.
- **Regime de medição**: 20 execuções de aquecimento descartadas, seguidas de
  300 medições cronometradas para o caso *single* (1 documento) e 60 para o
  caso *batch* (32 documentos — `max(300 // 5, 20)`), ambas ordenadas para o
  cálculo de percentis.
- **Concorrência de threads**: os dois backends rodam com
  `intra_op_num_threads=1` (explícito no `OnnxSessionOptions` do
  `OnnxPredictor`; o backend sklearn não usa paralelismo interno na predição
  de um `LogisticRegression`/TF-IDF esparso, então o efeito é equivalente) —
  para que a comparação isole o custo por requisição em vez de medir quanto
  paralelismo cada biblioteca consegue arrancar da máquina, cenário que não
  reflete o serviço em produção atendendo várias requisições concorrentes.
- **Paridade**: calculada sobre o held-out **completo** (2.770 documentos,
  não uma amostra de 500) — concordância de classe e diferença absoluta
  máxima de probabilidade entre os dois backends.

Reproduzir:

```bash
make benchmark
cat docs/latency_results.json | python -m json.tool
```

## 2. Baseline HTTP (Etapa 1, Task 7)

Antes de qualquer otimização de modelo, a Etapa 1 mediu a latência **fim a
fim via HTTP**, em Docker, sob WSL2 — 50 requisições `curl` contra
`POST /predict` do serviço já publicado como imagem:

| Métrica | Valor |
|---|---|
| p50 | 6.4 ms |
| p95 | 8.4 ms |
| p99 | 8.9 ms |
| `X-Process-Time-Ms` de uma amostra | 4.139 ms |

Esse baseline inclui o custo de rede local, parsing HTTP, (de)serialização
JSON e todo o middleware do FastAPI — não apenas a inferência do modelo. A
diferença entre o p50 de 6.4 ms e a amostra de processamento interno de
4.139 ms (~2.3 ms) é overhead de HTTP/serialização que nenhuma otimização de
modelo, incluindo a troca para ONNX, é capaz de reduzir. Comparar esse número
diretamente com os tempos de `predictor.predict()` medidos abaixo
superestimaria o ganho real percebido pelo cliente da API — por isso a
tabela da seção 3 mede apenas a chamada ao `predictor`, sem o transporte
HTTP, e a seção 5 volta a discutir explicitamente o que esse recorte deixa
de fora.

## 3. Tabela comparativa

Medido nesta máquina, `make benchmark`, execução de
2026-09-11 (arquivo completo em `docs/latency_results.json`):

### Single request (1 documento)

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | média (ms) | throughput (req/s) | artefato |
|---|---|---|---|---|---|---|
| sklearn | 0.80 | 1.09 | 1.45 | 0.83 | 1 202 | 6.0 MB (`pipeline.joblib`) |
| onnx | 0.26 | 0.34 | 0.45 | 0.27 | 3 726 | 2.7 MB (`pipeline.onnx`) |
| **speedup (p50)** | **3.13x** | 3.21x | 3.20x | 3.10x | 3.10x | — |

### Batch (32 documentos)

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | média (ms) | throughput (req/s) |
|---|---|---|---|---|---|
| sklearn | 6.83 | 8.55 | 9.36 | 7.06 | 4 534 |
| onnx | 12.85 | 15.11 | 27.14 | 13.00 | 2 462 |
| **speedup (p50)** | **0.53x (ONNX mais lento)** | 0.57x | 0.34x | 0.54x | 0.54x |

Os números de `p50_ms`/`p95_ms`/`p99_ms`/`mean_ms`/`throughput_rps` vêm
diretamente de `docs/latency_results.json`; os tamanhos de artefato citados
(MB) são a conversão dos bytes armazenados nesse mesmo arquivo. Execuções
repetidas do benchmark nesta máquina reproduzem o mesmo padrão qualitativo
(ganho de ~3x no single, perda no batch de 32) com variação de poucos
décimos de milissegundo entre rodadas.

## 4. Paridade

Sobre os 2.770 documentos do held-out:

- **Concordância de classe**: 100,00% (2770/2770).
- **Diferença absoluta máxima de probabilidade**: 2.813e-07.

Esse número bate com o medido de forma independente na Task 12 (2.813e-07,
100% de concordância) e com a checagem em amostra de 500 documentos feita
naquela mesma task (1.921e-07) — consistente com o esperado: mais documentos
tendem a incluir algum outlier ligeiramente pior, mas a ordem de grandeza não
muda. Ambos ficam muito abaixo da tolerância de 1e-4 usada nos testes de
paridade (`tests/inference/test_onnx_predictor.py`). O ganho de latência do
ONNX no caso single não é comprado com uma mudança de comportamento do
modelo.

## 5. Análise

**De onde vem o ganho, no caso single-request.** O grafo ONNX funde
vetorização TF-IDF e classificação em uma única execução dentro do ONNX
Runtime, em C++, sem retornar ao interpretador Python entre as duas etapas.
O `Pipeline` sklearn, por contraste, paga o custo de duas chamadas Python
distintas (`tfidf.transform` e `clf.predict_proba`), a criação de objetos
intermediários (a matriz esparsa do TF-IDF) e a checagem de tipos que o
scikit-learn faz a cada chamada. Para uma única string de entrada, esse
overhead fixo domina o tempo total — daí o ganho de ~3,1x observado.

**Onde o ganho desaparece — e se inverte.** No caso batch (32 documentos), o
resultado se inverte: o sklearn fica *mais rápido* que o ONNX (0,53x, ou
seja, o ONNX leva quase o dobro do tempo). A causa provável é estrutural, não
um artefato de medição: `scipy.sparse` implementa a multiplicação
matriz-esparsa × matriz-densa (o produto TF-IDF × pesos da regressão
logística) com rotinas nativas otimizadas especificamente para esse padrão
de dado, enquanto o grafo ONNX exportado por `skl2onnx`, ao converter o
`TfidfVectorizer`/`TfidfTransformer`, no fluxo usado aqui (necessário para
paridade em escala real de vocabulário — ver `src/triage/features/text.py`)
não preserva a mesma representação esparsa do sklearn ao longo de todo o
grafo; parte do trabalho passa a ser feito em cima de tensores densos ou de
operações não vetorizadas por lote, que crescem linearmente com o batch sem
o mesmo aproveitamento das rotinas de baixo nível do scipy/BLAS que o
sklearn usa. O resultado prático é visível no p99 sob a carga maior: passa
de 9,36 ms (sklearn) para 27,14 ms (onnx) — o pior caso do ONNX em batch é
quase 3x pior que o do sklearn, não apenas equivalente.

**Consequência para o serviço real.** O endpoint `POST /predict` do serviço
atende uma requisição por chamada — exatamente o caso single onde o ONNX
ganha por larga margem. O endpoint `POST /predict/batch` recebe lotes
definidos pelo cliente; para lotes do tamanho testado (32), o backend sklearn
é a escolha mais rápida. Não medimos o comportamento em lotes maiores ou
menores que 32 — a linha exata onde a vantagem se inverte fica como trabalho
futuro caso o tráfego de `/predict/batch` se torne relevante.

**O que este relatório não mede.** A tabela da seção 3 mede apenas
`predictor.predict()`, sem o transporte HTTP. O baseline da seção 2 mostra
que ~2,3 ms de overhead de rede/serialização não são tocados por nenhuma
troca de backend — então, na prática, o ganho percebido ponta a ponta pelo
cliente da API no caso single é menor, em termos absolutos, do que o fator
3,1x sugere (a diferença absoluta de ~0,55 ms entre os backends é pequena
frente aos ~6,4 ms do p50 fim a fim medido na Task 7); o fator 3,1x é real,
mas descreve apenas a fatia de inferência, não a resposta completa ao
cliente.

## 6. Decisão

**Backend padrão em produção: `onnx`, para o endpoint de predição
individual (`/predict`).** É o caminho que a maioria do tráfego esperado do
serviço percorre (uma triagem por vez, no momento do atendimento), e é onde
o ONNX efetivamente reduz a latência sem alterar as predições do modelo
(paridade de 100% de concordância, diferença de probabilidade de
2.813e-07).

Para o endpoint de lote (`/predict/batch`), com base na medição em batches
de 32, o backend sklearn seria a escolha mais rápida — mas o serviço não
expõe hoje uma forma de escolher o backend por endpoint (`TRIAGE_BACKEND` é
uma única variável de ambiente para todo o processo, ver
`src/triage/config/settings.py`); mudar isso é uma extensão possível, não
implementada nesta entrega. Como o caso de uso central (`/predict`,
requisição única) favorece claramente o ONNX e é o cenário mais representativo
do produto (triagem no momento do atendimento, não em lote), o padrão do
serviço permanece `onnx`, documentando aqui a ressalva sobre lotes para quem
for operar `/predict/batch` em volume.
