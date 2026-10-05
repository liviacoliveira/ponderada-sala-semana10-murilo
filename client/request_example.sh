#!/usr/bin/env bash
# Cliente de demonstração. Os valores abaixo são APENAS um exemplo (7 fechamentos diários).
# Substitua pelos últimos 7 fechamentos reais do seu CSV para uma demonstração realista.
set -e
URL=${URL:-http://localhost:8000}

echo "== /health =="
curl -s "$URL/health"; echo

echo "== /model-info =="
curl -s "$URL/model-info"; echo

echo "== /predict =="
curl -s -X POST "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"closes": [60000, 60500, 61200, 60800, 61500, 62000, 62300]}'; echo
