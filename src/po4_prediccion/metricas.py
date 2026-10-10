"""Métricas de evaluación e inferencia estadística.

* R², coeficiente de correlación de Pearson r, RMSE, MAE y sesgo.
* SS: índice de habilidad frente a persistencia, SS = 1 - MSE / MSE_persistencia.
* Intervalos de confianza por bootstrap de bloques móviles (bloque = 1 día).
* Prueba de Diebold-Mariano (pérdida cuadrática, varianza de Newey-West).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BLOQUE = 720  # orígenes por día (2 min)


def metricas(y, p, persistencia=None) -> dict[str, float]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0, 6)
    out = {
        "r2": r2_score(y, p),
        "r": float(np.corrcoef(y, p)[0, 1]),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "mae": mean_absolute_error(y, p),
        "sesgo": float(np.mean(p - y)),
    }
    if persistencia is not None:
        out["ss"] = 1 - mean_squared_error(y, p) / mean_squared_error(y, persistencia)
    return {k: float(v) for k, v in out.items()}


def bootstrap_r2(y, predicciones: dict[str, np.ndarray], rng, n_boot=1000) -> dict[str, np.ndarray]:
    """Distribución bootstrap del R² remuestreando bloques diarios contiguos."""
    n = len(y)
    inicios = rng.integers(0, n - BLOQUE, size=(n_boot, int(np.ceil(n / BLOQUE))))
    out = {k: np.empty(n_boot) for k in predicciones}
    for b in range(n_boot):
        idx = (inicios[b][:, None] + np.arange(BLOQUE)).ravel()[:n]
        yy = y[idx]
        sst = ((yy - yy.mean()) ** 2).sum()
        for k, p in predicciones.items():
            out[k][b] = 1 - ((yy - p[idx]) ** 2).sum() / sst
    return out


def diebold_mariano(e1, e2, rezago: int) -> tuple[float, float]:
    """H0: igual exactitud. Estadístico < 0 favorece al modelo 1."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    gamma = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, rezago + 1)]
    var = gamma[0] + 2 * sum((1 - k / (rezago + 1)) * gamma[k] for k in range(1, rezago + 1))
    est = d.mean() / np.sqrt(var / n)
    return float(est), float(2 * (1 - stats.norm.cdf(abs(est))))
