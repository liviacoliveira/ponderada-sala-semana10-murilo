"""Backend de inferência (FastAPI): carrega o modelo treinado e serve predições."""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/app")
from common.features import WINDOW, features_from_windows  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
state = {"bundle": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["bundle"] = joblib.load(MODEL_PATH)
        print(f"[backend] modelo carregado de {MODEL_PATH}")
    else:
        print(f"[backend] ATENÇÃO: {MODEL_PATH} não encontrado")
    yield


app = FastAPI(title="BTC next-day close predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    closes: List[float] = Field(
        ..., min_length=WINDOW,
        description=f"Últimos fechamentos diários (mais antigo → mais recente), mínimo {WINDOW}.",
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["bundle"] is not None}


@app.get("/model-info")
def model_info():
    if state["bundle"] is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    b = state["bundle"]
    return {"window": b["window"], "sklearn_version": b["sklearn_version"], "metrics": b["metrics"]}


@app.post("/predict")
def predict(req: PredictRequest):
    if state["bundle"] is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    if any(c <= 0 for c in req.closes):
        raise HTTPException(status_code=422, detail="Os preços devem ser positivos")
    window = np.array(req.closes[-WINDOW:], dtype=float)[None, :]
    ratio = float(state["bundle"]["model"].predict(features_from_windows(window))[0])
    last = float(window[0, -1])
    return {
        "last_close": last,
        "predicted_next_close": round(last * ratio, 2),
        "disclaimer": "Predição experimental; não é recomendação de investimento.",
    }
