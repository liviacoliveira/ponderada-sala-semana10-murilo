# DEVLOG — Atividade Ponderada M7 

Esse documento explica o passo a passo de todo o desenvolvimento da solução para a ponderada em questão, descrevendo as principais decisões tomadas e como elas foram de fato implementadas no projeto.

## Resumo da solução
Container `trainer` treina um Ridge sobre dados diários do BTC-USD e salva `model.joblib` no volume `./models`. Container `backend` (FastAPI) lê o artefato e expõe `/health`, `/model-info` e `/predict`. Um cliente (`curl`) consome a API.

## Fase 0 — Planejamento e UML 
**O que foi feito:** escolha da moeda/dado/horizonte, escolha do modelo, desenho da arquitetura.

**Decisões e justificativas**
| Decisão | Escolha | Por quê |
|---|---|---|
| Moeda e frequência | BTC-USD, diária | Exigência do tema; dado diário é leve e fácil de obter |
| Horizonte | Fechamento do dia seguinte | Alvo simples e bem definido |
| Fonte | Yahoo Finance via `yfinance` (CSV em `data/`) | Sem chave de API; gera CSV reutilizável |
| Modelo | Ridge sobre razões de preços (janela 7) | Simples, rápido, serializável; features invariantes à escala (preço do BTC muda muito ao longo do tempo) |
| Validação | Split cronológico 80/20 + baseline "amanhã = hoje" | Série temporal: não pode embaralhar; baseline dá referência honesta |
| Backend | FastAPI | Validação automática do payload e docs em `/docs` |
| Transferência do artefato | Volume Docker compartilhado `./models` | Simples e reprodutível; backend monta como somente leitura |

**Evidência:** `docs/arquitetura.png` (fonte: `docs/arquitetura.svg` / `.puml`).

## Fase 1 — Dados, treino e exportação ([hh:mm]–[hh:mm])
**O que foi feito:** `training/train.py` (carrega CSV ou baixa via yfinance, cria janelas de 7 dias, split cronológico, treina, avalia, salva `model.joblib` e `metrics.json`); `training/Dockerfile`.

**Comandos**
```bash
docker compose build trainer
docker compose run --rm trainer
```

**Resultado observado:** [cole aqui a saída do treino: nº de linhas, período, métricas]

| Métrica (teste) | Modelo | Baseline |
|---|---|---|
| MAE | [ ] | [ ] |
| RMSE | [ ] | [ ] |

**Análise:** [o modelo ficou melhor/pior/igual ao baseline? por quê? (preço diário ≈ passeio aleatório)]

**Evidência:** [print em `docs/evidencias/01-treino.png`; `models/model.joblib` gerado]

## Fase 2 — Backend de inferência ([hh:mm]–[hh:mm])
**O que foi feito:** `backend/app.py` carrega o artefato no startup; rotas `/health`, `/model-info`, `/predict`; validação (mínimo de 7 preços, valores positivos); `backend/Dockerfile`; `docker-compose.yml` com volume compartilhado, `depends_on` e healthcheck.

**Decisão importante:** o código de features está em `common/features.py` e é copiado para os dois containers, evitando divergência entre treino e inferência. Versões de numpy/scikit-learn/joblib fixadas igualmente nos dois `requirements.txt` (modelos pickled/joblib dependem da versão da biblioteca).

**Comandos**
```bash
docker compose up --build -d
docker compose ps
docker compose logs backend
```

**Resultado observado:** [cole `docker compose ps` e logs mostrando "modelo carregado"]

**Evidência:** [`docs/evidencias/02-compose-ps.png`]

## Fase 3 — Integração e testes
| # | Teste | Comando | Esperado | Obtido |
|---|---|---|---|---|
| 1 | Serviço ativo | `curl localhost:8000/health` | `{"status":"ok","model_loaded":true}` | [ ] |
| 2 | Info do modelo | `curl localhost:8000/model-info` | métricas e versão | [ ] |
| 3 | Predição válida | `bash client/request_example.sh` | JSON com `predicted_next_close` | [ ] |
| 4 | Payload inválido (poucos preços) | `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"closes":[1,2]}'` | HTTP 422 | [ ] |
| 5 | Backend sem modelo | [opcional: apagar `models/model.joblib` e reiniciar] | `/health` com `model_loaded:false`; `/predict` 503 | [ ] |

**Evidência:** [`docs/evidencias/03-predict.png`]

## Dificuldades e como foram resolvidas
| Problema | Causa | Solução |
|---|---|---|
| [ex.: erro ao baixar dados no container] | [ex.: sem internet / mudança de formato do yfinance] | [ex.: baixar CSV manualmente e colocar em `data/`] |
| [ ] | [ ] | [ ] |

## Alterações ao longo do desenvolvimento
- [hh:mm] [o que mudou e por quê]

## Limitações conhecidas
- Apenas preço histórico como entrada; mercado cripto é altamente ruidoso, e o modelo tende a ficar próximo do baseline.
- Sem retreino automático, autenticação ou monitoramento.
- Avaliação em um único split temporal.
- Predições experimentais; **não** são recomendação de investimento.

## Conclusão
[2–3 frases: o que funcionou, o que aprendeu, próximos passos possíveis (mais features, outros modelos, validação walk-forward, retreino agendado)]
