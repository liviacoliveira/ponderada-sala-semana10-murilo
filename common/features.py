"""Features compartilhadas entre treino e inferência (mesmo código nos dois containers).

Janela de WINDOW fechamentos consecutivos (do mais antigo ao mais recente).
Features = razões close_{t-k} / close_t, k = 1..WINDOW-1  (invariantes à escala do preço).
Alvo      = close_{t+1} / close_t  (razão do dia seguinte).
Predição  = close_t * razão_prevista.
"""
import numpy as np

WINDOW = 7


def features_from_windows(windows: np.ndarray) -> np.ndarray:
    """windows: array (n, WINDOW). Retorna array (n, WINDOW-1)."""
    windows = np.asarray(windows, dtype=float)
    last = windows[:, -1:]
    return windows[:, :-1] / last
