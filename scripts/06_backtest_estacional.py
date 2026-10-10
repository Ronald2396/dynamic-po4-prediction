"""Paso 6. Validación por origen móvil en cuatro trimestres (sección 4.6).

Para cada trimestre (ago-2022 a jul-2023) se reentrenan la MLP (una semilla,
versión conservadora) y LightGBM con todos los datos anteriores. Se reporta
el R² por trimestre y el R² agregado del año con IC 95 % (bootstrap diario).
"""

import time

import numpy as np
import pandas as pd

from po4_prediccion.datos import (FOLDS_ESTACIONALES, HORIZONTES, OBJETIVO, SEMILLA, TABLAS, asegurar_directorios,
                                  columnas_entrada, construir_dataset, particion_fold)
from po4_prediccion.metricas import bootstrap_r2, metricas
from po4_prediccion.red_neuronal import entrenar_mlp
from po4_prediccion.referencias import lightgbm_delta

asegurar_directorios()
t0 = time.time()
crudo, f = construir_dataset()
X = f[columnas_entrada(f)].to_numpy("float32")
po = crudo[OBJETIVO].to_numpy()
hs = list(HORIZONTES)
Y = np.column_stack([f[f"target_{h}"].to_numpy() for h in hs])
D = Y - po[:, None]
NOMBRES = ("Persistencia", "LightGBM", "MLP propuesta")
filas, acumulado = [], {h: {k: [] for k in ("y",) + NOMBRES} for h in hs}
for inicio, fin in FOLDS_ESTACIONALES:
    tr, va, te = particion_fold(crudo["date"], inicio, fin, max(HORIZONTES.values()))
    P = entrenar_mlp(X, D, tr, va, semilla=SEMILLA).predecir_delta(X[te]) + po[te, None]
    for j, (h, pasos) in enumerate(HORIZONTES.items()):
        trh, vah, teh = particion_fold(crudo["date"], inicio, fin, pasos)
        lgb_full = lightgbm_delta(X, Y[:, j], po, trh, vah, te)
        for nombre, p in (("Persistencia", po[te]), ("LightGBM", lgb_full), ("MLP propuesta", P[:, j])):
            p = np.clip(p, 0, 6)
            filas.append({"horizonte_min": h, "trimestre": f"{inicio}..{fin}", "modelo": nombre,
                          **metricas(Y[te, j], p, po[te])})
            acumulado[h][nombre].append(p)
        acumulado[h]["y"].append(Y[te, j])
    print(inicio, round((time.time() - t0) / 60, 1), "min", flush=True)

rng = np.random.default_rng(SEMILLA)
agregado = []
for h in hs:
    y = np.concatenate(acumulado[h].pop("y"))
    pp = {k: np.concatenate(v) for k, v in acumulado[h].items()}
    bs = bootstrap_r2(y, pp, rng)
    for k in NOMBRES:
        agregado.append({"horizonte_min": h, "modelo": k, **metricas(y, pp[k], pp["Persistencia"]),
                         "r2_ic_inf": np.quantile(bs[k], .025), "r2_ic_sup": np.quantile(bs[k], .975)})
res = pd.DataFrame(filas)
res.to_csv(TABLAS / "06_backtest_trimestres.csv", index=False)
pd.DataFrame(agregado).to_csv(TABLAS / "06_backtest_agregado.csv", index=False)
print(res.pivot_table(index=["horizonte_min", "modelo"], columns="trimestre", values="r2").round(3).to_string())
print(pd.DataFrame(agregado)[["horizonte_min", "modelo", "r2", "r2_ic_inf", "r2_ic_sup", "rmse"]].round(3).to_string())
