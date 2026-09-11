# Model card — classificador de triagem de laudos médicos

## Propósito e público

Classifica o resumo (abstract) de um laudo médico em uma de cinco condições
clínicas e deriva, a partir dessa condição, um nível de urgência de triagem
(`normal`, `atencao`, `urgente`). O público-alvo é uma equipe de engenharia
ou produto avaliando o sistema como prova de conceito de apoio à
priorização de atendimento — **não** é destinado a uso clínico direto sem
supervisão humana. Ver seção de limitações abaixo.

## Dados

**Medical Abstracts TC Corpus** — abstracts de artigos científicos da área
médica, rotulados em 5 classes de condição:

| Rótulo | Condição |
|---|---|
| 1 | neoplasms |
| 2 | digestive system diseases |
| 3 | nervous system diseases |
| 4 | cardiovascular diseases |
| 5 | general pathological conditions |

- **Treino** (`medical_tc_train.csv`): usado para treino e validação, com
  split estratificado 80/20 (`triage.data.loader.train_val_split`,
  `SEED=42`).
- **Teste/held-out** (`medical_tc_test.csv`): 2.888 linhas brutas; após
  limpeza (remoção de abstracts vazios e duplicados,
  `triage.data.loader.clean`), 2.770 documentos — é sobre este conjunto
  limpo que todas as métricas de held-out deste documento e do benchmark de
  latência (`docs/latency_report.md`) são calculadas.
- Os CSVs não são versionados no repositório (`.gitignore`); ver instruções
  de obtenção no `README.md`.

## Pré-processamento

- Normalização feita pelo próprio `TfidfVectorizer`: minúsculas,
  `token_pattern=r"[a-z]{2,}"` (mantém apenas sequências de letras com 2+
  caracteres, descarta dígitos e pontuação).
- **Sem remoção de stop words.** Essa é uma mudança deliberada em relação a
  uma versão anterior do pipeline: `stop_words="english"` combinado com
  bigramas não é seguro de converter para ONNX com paridade aceitável (ver
  `src/triage/features/text.py`) — a ordem em que o scikit-learn remove
  stop words antes de montar os n-gramas não é reproduzida pelo conversor.
  Essa é a causa direta da queda de desempenho descrita na seção de
  métricas abaixo.
- N-gramas de 1 e 2 tokens, `min_df=2`, `max_df=0.9`, `max_features=50 000`,
  `sublinear_tf=True`.
- `TraceableTfidfVectorizer` (subclasse de `TfidfVectorizer`, de
  `skl2onnx.sklapi`) no lugar do `TfidfVectorizer` padrão — necessária para
  que a decomposição de n-gramas sobreviva à conversão ONNX na escala real
  de vocabulário (ver limitações).

## Arquitetura do modelo

**Regressão logística linear** (`sklearn.linear_model.LogisticRegression`,
`class_weight="balanced"`) sobre features TF-IDF esparsas.

A escolha por um modelo linear não é apenas de desempenho — é uma restrição
de compatibilidade: o pipeline precisa ser exportável para ONNX
(`skl2onnx`) para viabilizar o backend de inferência de baixa latência (ver
`docs/latency_report.md`), e modelos lineares sobre TF-IDF são bem
suportados por esse conversor, com paridade numérica verificável. A
comparação de candidatos (`scripts/train.py --compare`, ver tabela abaixo)
confirma que essa escolha também é competitiva em qualidade:

| Modelo | macro-F1 (validação) | acurácia (validação) |
|---|---|---|
| logistic_regression | 0.6847 | — |
| linear_svc | 0.6843 | — |
| naive_bayes | 0.4043 | — |
| dummy (baseline trivial) | 0.0978 | — |

`logistic_regression` e `linear_svc` (SVM linear calibrado) empatam
tecnicamente (diferença de macro-F1 de 0.0004); `logistic_regression` foi
escolhido por ser o mais simples dos dois modelos competitivos — sem a
camada extra de calibração (`CalibratedClassifierCV`) que `linear_svc`
precisa só para expor `predict_proba`.

## Métricas no held-out

Held-out (`medical_tc_test.csv`, 2.770 documentos após limpeza):

- **macro-F1**: 0.5900
- **acurácia**: 0.5874

Esses números são ~1.3 pp abaixo do que o mesmo pipeline alcançava antes da
mudança de vetorizador exigida pela conversão ONNX (held-out 0.6030,
validação 0.6950) — a remoção de `stop_words` custou desempenho
mensurável, e este documento a reporta sem atenuar: é o preço real pago
para viabilizar a otimização de latência descrita em
`docs/latency_report.md`, uma troca deliberada, não um efeito colateral
escondido.

### Métricas por classe (held-out)

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| neoplasms | 0.680 | 0.744 | 0.710 | 616 |
| digestive system diseases | 0.485 | 0.641 | 0.553 | 287 |
| nervous system diseases | 0.526 | 0.637 | 0.576 | 366 |
| cardiovascular diseases | 0.653 | 0.722 | 0.685 | 589 |
| general pathological conditions | 0.525 | 0.359 | 0.426 | 912 |

`general pathological conditions` é de longe a classe com pior F1
(0.426) apesar de ter o maior suporte (912) — é a classe mais heterogênea
por definição (um "catch-all" de condições patológicas gerais), o que a
torna a mais difícil de separar linearmente das demais; a matriz de
confusão abaixo confirma que ela concentra a maior parte dos erros de
classificação cruzada.

### Matriz de confusão (held-out)

Linhas = rótulo verdadeiro, colunas = predito, ordem
`[neoplasms, digestive, nervous, cardiovascular, general]`:

```
                    neo   dig   nerv  card  gen
neoplasms           458    41    39    15    63
digestive            34   184     8     9    52
nervous              25     8   233    23    77
cardiovascular       14    15    31   425   104
general             143   131   132   179   327
```

## Limitações

- **Descasamento de domínio**: o corpus é composto de abstracts de artigos
  científicos em inglês, escritos para uma audiência acadêmica — não são
  laudos hospitalares reais, não seguem o formato, o vocabulário nem o
  registro linguístico de um prontuário ou de uma nota de pronto-socorro.
  Um modelo treinado neste corpus não deve ser assumido como diretamente
  aplicável a texto clínico real sem validação adicional nesse domínio.
- **Urgência é heurística de produto, não rótulo clínico**: o mapa de
  condição → urgência (`src/triage/urgency.py`) foi definido para atender
  ao requisito funcional de "nível de urgência" do enunciado do desafio. Não
  é derivado de nenhum protocolo de triagem clínica validado (como Manchester
  ou ESI), não foi revisado por profissional de saúde, e não deve ser tratado
  como equivalente a uma classificação de gravidade clinicamente calibrada.
- **Ferramenta de apoio, não substituta**: o sistema prioriza e sugere; a
  decisão de atendimento permanece com o profissional de saúde. Nenhuma
  predição deste modelo deve determinar, sozinha, a condução clínica de um
  caso.
- **Acoplamento com `skl2onnx` no artefato persistido**: `build_vectorizer`
  retorna um `TraceableTfidfVectorizer`, classe definida em
  `skl2onnx.sklapi`. Isso significa que `pipeline.joblib` embute uma
  referência a essa classe, e **qualquer processo que precise desserializar
  o artefato — incluindo um deployment que só use o backend sklearn e nunca
  rode uma conversão ONNX — precisa ter `skl2onnx` instalado**, só para o
  `joblib.load` funcionar. Essa dependência foi aceita deliberadamente: é o
  que permite atingir a paridade sklearn↔ONNX medida em
  `docs/latency_report.md` (2.813e-07 de diferença máxima de probabilidade)
  na escala real de vocabulário do projeto (`max_features=50 000`); a
  alternativa (`TfidfVectorizer` comum) chegava a ~0.12 de diferença e ~96%
  de concordância de classe, inaceitável para uma otimização que não pode
  mudar o comportamento do modelo. O custo é real e teria impacto direto em
  qualquer estratégia de redução de tamanho de imagem de produção (ver
  `docs/deploy_architecture.md`).
- **Desempenho desigual entre classes**: como a tabela por classe mostra,
  `general pathological conditions` tem F1 substancialmente pior que as
  demais classes — um viés estrutural do problema (classe mais heterogênea),
  não um artefato de amostragem, que qualquer uso do modelo deveria levar em
  conta ao interpretar suas predições para essa categoria especificamente.

## Considerações éticas e de viés

- O corpus reflete a distribuição de tópicos e a linguagem de literatura
  médica publicada, que não é demograficamente neutra nem representativa de
  todas as populações de pacientes — vieses de quem produz e é objeto dessa
  literatura podem se propagar ao modelo sem que o pipeline atual meça ou
  corrija isso.
- O mapa de urgência, por ser heurística de produto, carrega os julgamentos
  de quem o definiu, não um consenso clínico auditável — decisões de
  produto que usem esse mapa para priorizar atendimento devem estar cientes
  de que ele não tem a mesma base de evidência que uma escala de triagem
  validada.
- Erros de classificação — em particular, na classe mais fraca (`general
  pathological conditions`) — podem levar a uma priorização de urgência
  equivocada; o sistema deve ser posicionado, em qualquer uso real, como
  apoio a uma decisão humana e nunca como decisão automática de
  atendimento.
