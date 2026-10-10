"""Modelos de referencia: persistencia, naive de ciclo previo, modelo
mecanístico reducido conmutado por fase y LightGBM.

Modelo mecanístico (balance de primer orden inspirado en ASM2d):

    dS/dt = k_m (S*_m - S),   S*_m = b_m * S_media_3h + c_m,
    m ∈ {alimentación, sin alimentación}

El calendario de fases futuro se proyecta de forma causal a partir de la
duración de las últimas fases completas. Integración de Euler (paso 2 min).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .datos import OBJETIVO, SEMILLA

PARAMETROS_MECANISTICO = ["k_sin_alim", "k_alim", "b_sin_alim", "b_alim", "c_sin_alim", "c_alim"]
PARAMETROS_LGB = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbose=-1, seed=SEMILLA, num_threads=16)


def persistencia(crudo: pd.DataFrame) -> np.ndarray:
    return crudo[OBJETIVO].to_numpy()


def naive_ciclo_previo(crudo: pd.DataFrame, f: pd.DataFrame, pasos: int) -> np.ndarray:
    """PO4 en t + h - L (misma posición del ciclo anterior). Si L < h se usaría
    un valor futuro, por lo que se limita a t (causal)."""
    po = crudo[OBJETIVO].to_numpy()
    L = f["last_cycle_len"].fillna(40).clip(5, 300).to_numpy().astype(int)
    t = np.arange(len(crudo))
    idx = np.clip(np.minimum(t + pasos - L, t), 0, None)
    return po[idx]


def calendario_fases(f: pd.DataFrame, pasos: int) -> np.ndarray:
    """Indicador proyectado de alimentación para k = 1..pasos (N x pasos)."""
    actual = (f["inl"].to_numpy() == 2).astype(int)
    e = f["since_inl_change"].to_numpy()
    l_igual = f["prev_inl_run2"].fillna(19).clip(2, 200).to_numpy()
    l_otra = f["prev_inl_run"].fillna(19).clip(2, 200).to_numpy()
    resto = np.maximum(l_igual - e - 1, 0)
    k = np.arange(1, pasos + 1)[None, :]
    j = np.mod(k - resto[:, None] - 1, (l_otra + l_igual)[:, None])
    en_actual = (k <= resto[:, None]) | (j >= l_otra[:, None])
    return np.where(en_actual, actual[:, None], 1 - actual[:, None])


def simular_mecanistico(p, S0, S_media, calendario):
    a = np.array([p[0], p[1]]); b = np.array([p[2], p[3]]); c = np.array([p[4], p[5]])
    S = S0.copy()
    for k in range(calendario.shape[1]):
        m = calendario[:, k]
        S = S + a[m] * (b[m] * S_media + c[m] - S)
    return np.clip(S, 0, 6)


def ajustar_mecanistico(S0, S_media, calendario, y, idx):
    x0 = np.array([0.05, 0.05, 0.6, 1.6, 0.0, 0.2])
    r = least_squares(lambda p: simular_mecanistico(p, S0[idx], S_media[idx], calendario[idx]) - y[idx], x0,
                      bounds=([0, 0, 0, 0, -2, -2], [1, 1, 5, 5, 2, 2]), max_nfev=200)
    return r.x


def lightgbm_delta(X, y, base, tr, va, te):
    """LightGBM sobre el cambio y(t+h) - y(t); detención temprana en validación."""
    dtr = lgb.Dataset(X[tr], y[tr] - base[tr])
    dva = lgb.Dataset(X[va], y[va] - base[va], reference=dtr)
    m = lgb.train(PARAMETROS_LGB, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(X[te], num_iteration=m.best_iteration) + base[te]
