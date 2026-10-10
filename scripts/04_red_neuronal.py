"""Paso 4. Entrenamiento de la red neuronal propuesta (sección 4.4).

Cinco MLP multihorizonte con distinta semilla. El modelo propuesto es el
ensamble (promedio) de las cinco; también se guarda el desempeño individual
de cada semilla y las curvas de entrenamiento.
"""

import time

import numpy as np
import pandas as pd

from po4_prediccion.datos import (HORIZONTES, MODELOS, OBJETIVO, PREDICCIONES, TABLAS, asegurar_directorios,
                                  columnas_entrada, construir_dataset, particion_cronologica)
from po4_prediccion.metricas import metricas
from po4_prediccion.red_neuronal import SEMILLAS, entrenar_mlp

asegurar_directorios()
t0 = time.time()
crudo, f = construir_dataset()
cols = columnas_entrada(f)
X = f[cols].to_numpy("float32")
po = crudo[OBJETIVO].to_numpy()
hs = list(HORIZONTES)
Y = np.column_stack([f[f"target_{h}"].to_numpy() for h in hs])
D = Y - po[:, None]  # objetivo: cambio de PO4
tr, va, te = particion_cronologica(len(f), max(HORIZONTES.values()))

filas, historial, predicciones = [], [], []
for s in SEMILLAS:
    red = entrenar_mlp(X, D, tr, va, semilla=s)
    P = red.predecir_delta(X[te]) + po[te, None]
    predicciones.append(P)
    for j, h in enumerate(hs):
        filas.append({"semilla": s, "horizonte_min": h, "epocas": len(red.historial),
                      **metricas(Y[te, j], P[:, j], po[te])})
    historial += [{"semilla": s, **r} for r in red.historial]
    red.guardar(MODELOS / f"mlp_semilla_{s}.pt", cols)
    print("semilla", s, round((time.time() - t0) / 60, 1), "min", flush=True)

ens = np.clip(np.mean(predicciones, 0), 0, 6)
salida = {"y": Y[te].astype("float32")}
for j, h in enumerate(hs):
    salida[f"{h}|MLP propuesta"] = ens[:, j]
    salida[f"{h}|MLP individual (semilla 42)"] = np.clip(predicciones[0][:, j], 0, 6)
np.savez_compressed(PREDICCIONES / "red_neuronal.npz", **salida)
sem = pd.DataFrame(filas)
sem.to_csv(TABLAS / "04_metricas_por_semilla.csv", index=False)
pd.DataFrame(historial).to_csv(TABLAS / "04_curvas_entrenamiento.csv", index=False)
print(sem.groupby("horizonte_min")[["r2", "rmse", "mae"]].agg(["mean", "std"]).round(4).to_string())
for j, h in enumerate(hs):
    print(h, "ensamble", {k: round(v, 4) for k, v in metricas(Y[te, j], ens[:, j], po[te]).items()})
