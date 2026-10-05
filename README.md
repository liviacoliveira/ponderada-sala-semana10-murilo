# Atividade Ponderada M7 

*OBS: o diário de desenvolvimento está em [DEVLOG.md](DEVLOG.md), que evidencia todo o processo de desenvolvimento e tomada de decisão desse projeto em questão.

Solução conteinerizada: um container treina um modelo a partir de dados históricos do BTC-USD e grava o artefato em um volume; um segundo container*backend (FastAPI/Python) carrega o artefato e responde predições do fechamento do dia seguinte.


![Arquitetura](docs/arquitetura.png)

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

## Decisões principais
- **Dado:** BTC-USD diário (`Date`, `Close`), a partir de 2018. **Horizonte:** fechamento do dia seguinte.
- **Features:** razões `close_{t-k}/close_t` (k=1..6) em janela de 7 dias, invariantes à escala do preço. Alvo: `close_{t+1}/close_t`.
- **Modelo:** Ridge (regressão linear regularizada), escolhido pela simplicidade; comparado a um baseline "amanhã = hoje".
- **Split:** cronológico (80% treino / 20% teste), sem embaralhar.
- **Entrega do artefato:** volume `./models` (escrita no trainer, somente leitura no backend); `depends_on: service_completed_successfully` garante a ordem.
- **Mesmas versões** de numpy/scikit-learn/joblib nos dois containers e mesmo código de features (`common/features.py`).

## Endpoints
| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Serviço ativo e se o modelo foi carregado |
| GET | `/model-info` | Metadados e métricas do modelo |
| POST | `/predict` | Body `{"closes":[...≥7 valores]}` → `predicted_next_close` |

## Limitações
- Usa apenas preço histórico; mercado cripto é muito ruidoso e o modelo tende a ficar próximo do baseline.
- Sem retreino automático, sem autenticação, sem monitoramento.
- Avaliação em um único split temporal.

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
