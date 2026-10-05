# Atividade Ponderada M7 

Esse README serve apenas como um guia de execução do projeto. O **diário de desenvolvimento** está em [DEVLOG.md](DEVLOG.md), que evidencia todo o processo de desenvolvimento e tomada de decisão desse projeto em questão.

## Como reproduzir

Pré-requisito: Docker + Docker Compose.

```bash
# 1. Treinar (baixa o CSV via yfinance se data/btc_usd_daily.csv não existir) e subir o backend
docker compose up --build -d

# 2. Verificar
curl localhost:8000/health
curl localhost:8000/model-info

# 3. Pedir uma predição (últimos 7+ fechamentos, do mais antigo ao mais recente)
bash client/request_example.sh
```

Para apenas retreinar: `docker compose run --rm trainer`. Para ver logs: `docker compose logs trainer backend`.

Alternativa sem internet no container: coloque um CSV com colunas `Date,Close` em `data/btc_usd_daily.csv` (ex.: Yahoo Finance ou CryptoDataDownload).

## Endpoints
| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Serviço ativo e se o modelo foi carregado |
| GET | `/model-info` | Metadados e métricas do modelo |
| POST | `/predict` | Body `{"closes":[...≥7 valores]}` → `predicted_next_close` |


## Estrutura
```
common/features.py    features compartilhadas
training/             train.py, Dockerfile, requirements
backend/              app.py (FastAPI), Dockerfile, requirements
client/               exemplo de requisições
docs/                 UML (png, svg, puml) e evidencias/
models/               artefato gerado (model.joblib, metrics.json)
DEVLOG.md             diário de desenvolvimento
```
