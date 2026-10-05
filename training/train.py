"""Treina um modelo simples para prever o fechamento do dia seguinte do BTC-USD."""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/app")  # caminho dentro do container
from common.features import WINDOW, features_from_windows  # noqa: E402

DATA_PATH = Path(os.getenv("DATA_PATH", "data/btc_usd_daily.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
METRICS_PATH = MODEL_PATH.with_name("metrics.json")
TEST_FRAC = float(os.getenv("TEST_FRAC", "0.2"))


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"[train] {DATA_PATH} não encontrado; baixando via yfinance...")
        import yfinance as yf

        raw = yf.download("BTC-USD", start="2018-01-01", interval="1d", auto_adjust=True, progress=False)
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw.reset_index()[["Date", "Close"]]
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    return df


def main() -> None:
    df = load_data()
    print(f"[train] {len(df)} linhas, de {df['Date'].min().date()} até {df['Date'].max().date()}")

    close = df["Close"].to_numpy(dtype=float)
    # janelas de WINDOW dias; alvo = fechamento do dia seguinte à janela
    windows = sliding_window_view(close, WINDOW)[:-1]
    next_close = close[WINDOW:]
    last_close = windows[:, -1]

    X = features_from_windows(windows)
    y = next_close / last_close

    # split CRONOLÓGICO (sem embaralhar)
    split = int(len(X) * (1 - TEST_FRAC))
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]
    last_test, true_test = last_close[split:], next_close[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    pred_test = last_test * model.predict(X_test)
    baseline = last_test  # "amanhã = hoje"

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_mae": float(mean_absolute_error(true_test, pred_test)),
        "model_rmse": rmse(true_test, pred_test),
        "baseline_mae": float(mean_absolute_error(true_test, baseline)),
        "baseline_rmse": rmse(true_test, baseline),
    }
    print("[train] métricas (teste):", json.dumps(metrics, indent=2))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "window": WINDOW, "sklearn_version": sklearn.__version__, "metrics": metrics},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[train] modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
