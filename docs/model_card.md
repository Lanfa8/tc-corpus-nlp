# Model card — classificador de triagem de laudos médicos

## Propósito

Classifica o resumo (abstract) de um laudo médico em uma de cinco condições
clínicas e deriva, a partir dessa condição, um nível de urgência de triagem
(`normal`, `atencao`, `urgente`). É uma prova de conceito de apoio à
priorização de atendimento, **não** um sistema para uso clínico sem
supervisão humana.

## Dados

**Medical Abstracts TC Corpus** — abstracts de artigos científicos da área
médica, rotulados em 5 classes: neoplasms, digestive system diseases,
nervous system diseases, cardiovascular diseases e general pathological
conditions.

- **Treino** (`medical_tc_train.csv`): split estratificado 80/20 para
  treino e validação (`SEED=42`).
- **Held-out** (`medical_tc_test.csv`): 2.770 documentos após remoção de
  abstracts vazios e duplicados.

## Modelo

**TF-IDF + regressão logística** (`class_weight="balanced"`), com
n-gramas de 1 e 2 tokens, `max_features=50 000` e `sublinear_tf=True`.

Um modelo linear sobre TF-IDF é bem suportado pelo `skl2onnx`, o que
viabiliza a otimização para ONNX (ver `docs/latency_report.md`). A
comparação de candidatos (`scripts/train.py --compare`) mostra que a
escolha também é competitiva em qualidade:

| Modelo | macro-F1 (validação) | acurácia (validação) |
|---|---|---|
| logistic_regression | 0.6847 | 0.6866 |
| linear_svc | 0.6843 | 0.6940 |
| naive_bayes | 0.4043 | 0.5670 |
| dummy (baseline trivial) | 0.0978 | 0.3235 |

`logistic_regression` e `linear_svc` empatam em macro-F1; a regressão
logística foi escolhida por ser mais simples (não precisa de calibração
para expor `predict_proba`).

## Métricas no held-out

- **macro-F1**: 0.5900
- **acurácia**: 0.5874

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| neoplasms | 0.680 | 0.744 | 0.710 | 616 |
| digestive system diseases | 0.485 | 0.641 | 0.553 | 287 |
| nervous system diseases | 0.526 | 0.637 | 0.576 | 366 |
| cardiovascular diseases | 0.653 | 0.722 | 0.685 | 589 |
| general pathological conditions | 0.525 | 0.359 | 0.426 | 912 |

`general pathological conditions` é a classe mais fraca: é a mais
heterogênea (um "catch-all") e concentra a maior parte dos erros.

## Limitações

- **Domínio**: o corpus é de abstracts científicos em inglês, não de laudos
  hospitalares reais. O modelo precisaria de validação em texto clínico
  real antes de qualquer uso.
- **Urgência é heurística**: o mapa condição → urgência
  (`src/triage/urgency.py`) não vem de um protocolo clínico validado (como
  Manchester ou ESI) e não foi revisado por profissional de saúde.
- **Troca por compatibilidade com ONNX**: a remoção de `stop_words`,
  necessária para a paridade sklearn ↔ ONNX, custou ~1.3 pp de macro-F1 no
  held-out (0.6030 → 0.5900).
