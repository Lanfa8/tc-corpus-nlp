# Roteiro do vídeo (formato STAR, ~5 minutos)

Roteiro para a gravação exigida pela Etapa 4. Estrutura STAR (Situação,
Tarefa, Ação, Resultado), com marcações de tempo aproximadas.

## Antes de gravar — abas e serviços que precisam estar prontos

1. Terminal no diretório raiz do projeto, com `make up` já rodado
   (API + Prometheus + Grafana no ar) e os dados/modelo já treinados em
   `models/`.
2. Aba do navegador em `http://localhost:8000/docs` (Swagger UI da API).
3. Aba do navegador em `http://localhost:3000` (Grafana), já logado
   (`admin`/`admin`), com o dashboard "Triagem de Laudos — Visão Geral"
   aberto.
4. Aba do navegador em `http://localhost:9090` (Prometheus) — opcional, útil
   se quiser mostrar uma query crua além do dashboard.
5. `make airflow-up` já rodado, aba em `http://localhost:8080` logada
   (`admin`/`admin`), com a DAG `triage_training` visível — **confirmar que
   ela aparece como "Unpaused"**, não "Paused" (a correção
   `is_paused_upon_creation=False` no `@dag(...)` evita a armadilha de uma
   DAG nova nascer pausada e uma run disparada ficar presa em `queued`
   silenciosamente; se o Airflow do ambiente de gravação já tiver essa DAG
   registrada de uma execução anterior a esta correção, ela pode continuar
   pausada mesmo assim — a opção não é retroativa — então despause-a
   manualmente antes de gravar).
6. Aba do GitHub Actions do repositório, mostrando o workflow mais recente
   verde.
7. Um segundo terminal livre, para rodar `make load-test` e a troca de
   backend durante a gravação sem precisar alternar de janela.

## Comandos a rodar, na ordem

```bash
make up                 # já deve estar rodando — não precisa repetir ao vivo
                         # (backend padrão onnx; requer make export-onnx antes)
make load-test          # gera tráfego contra a API, com o Grafana aberto ao lado
# no Airflow: disparar (trigger) a DAG triage_training manualmente pela UI
# TRIAGE_BACKEND=onnx -> sklearn: `TRIAGE_BACKEND=sklearn docker compose up
# -d --build` para comparar com o backend não otimizado, observando o
# painel de latência por backend no Grafana antes e depois
```

## Situação (0:00–0:45)

- Contexto: um serviço de saúde recebe laudos médicos em texto livre e
  precisa priorizar quem é atendido primeiro.
- Triagem manual é lenta e inconsistente sob volume — o atraso em identificar
  um caso mais urgente tem custo real.
- Objetivo do projeto: um classificador automático de laudos em 5 condições
  clínicas, com um nível de urgência derivado, servido como API de baixa
  latência.

## Tarefa (0:45–1:30)

- Requisitos do desafio: uma API de inferência com baseline de latência
  medido; pipeline de CI (lint + testes) no GitHub Actions; orquestração de
  treino/retreino via Airflow; observabilidade via Prometheus + Grafana;
  treino e comparação de modelos, com uma otimização de latência mensurável
  (ONNX) ao final.
- Enfatizar o disclaimer: urgência aqui é heurística de apoio à priorização,
  não substitui avaliação de um profissional de saúde.

## Ação (1:30–3:30)

1. Mostrar o diagrama de arquitetura (`docs/deploy_architecture.md` — seção
   de arquitetura em nuvem, e o diagrama local do `README.md`).
2. `make up` (mencionar que já está rodando) — subir API, Prometheus e
   Grafana com um único comando.
3. Abrir `/docs` do FastAPI, mostrar os endpoints (`/predict`,
   `/predict/batch`, `/health`, `/model/info`, `/metrics`).
4. Fazer uma predição ao vivo pelo Swagger UI com um exemplo de abstract,
   mostrando a condição prevista e o nível de urgência derivado.
5. Rodar `make load-test` com o dashboard do Grafana visível ao lado,
   mostrando os painéis reagindo em tempo real (taxa de requisições,
   latência, contagem de predições por condição/urgência).
6. Trocar para o Airflow: mostrar a DAG `triage_training` (unpaused),
   disparar uma execução manual e acompanhar as tasks
   (`validate_data → load_data → train_model → evaluate_model →
   export_onnx → register_artifacts`) até o sucesso.
7. Trocar `TRIAGE_BACKEND=onnx` (padrão) → `sklearn` no serviço e mostrar,
   no painel de latência por backend do Grafana, a mudança refletida.

## Resultado (3:30–5:00)

- Tabela de latência antes/depois (de `docs/latency_report.md`): ganho de
  ~3,1x no p50 de requisição única com ONNX (0.80 ms → 0.26 ms), com a
  ressalva honesta de que o mesmo backend fica mais lento em lotes de 32
  documentos (6.83 ms → 12.85 ms) — nem toda otimização generaliza para
  todo padrão de uso, e o relatório documenta os dois lados.
- macro-F1 no held-out: 0.5900, acurácia 0.5874 — com a observação de que
  essa é uma queda de ~1.3 pp em relação à versão anterior à otimização
  ONNX, custo aceito conscientemente pela paridade viabilizada.
- CI verde no GitHub Actions (mostrar a aba já aberta).
- Lições aprendidas: compatibilidade com ONNX impôs restrições reais de
  pré-processamento (sem `stop_words`); a escolha de backend não é
  universalmente melhor — depende do padrão de tráfego (single vs. batch);
  uma DAG nova nasce pausada por padrão no Airflow, uma armadilha fácil de
  levar para produção sem perceber.
