"""Paso 8. Importancia por permutación agrupada (sección 4.8).

Todas las columnas de una familia de variables se permutan conjuntamente
(misma permutación de filas) y se mide la pérdida de R² del ensamble de MLP
en una muestra de 40 000 orígenes del conjunto de prueba (5 repeticiones).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from po4_prediccion.datos import (FAMILIAS, HORIZONTES, MODELOS, OBJETIVO, SEMILLA, TABLAS, asegurar_directorios,
                                  columnas_entrada, construir_dataset, familia_de, particion_cronologica)
from po4_prediccion.red_neuronal import SEMILLAS, Predictor

asegurar_directorios()
crudo, f = construir_dataset()
cols = columnas_entrada(f)
redes = []
for s in SEMILLAS:
    red, c = Predictor.cargar(MODELOS / f"mlp_semilla_{s}.pt")
    assert c == cols
    redes.append(red)
X = f[cols].to_numpy("float32")
po = crudo[OBJETIVO].to_numpy()
hs = list(HORIZONTES)
Y = np.column_stack([f[f"target_{h}"].to_numpy() for h in hs])
_, _, te = particion_cronologica(len(f), max(HORIZONTES.values()))
rng = np.random.default_rng(SEMILLA)
idx = np.sort(rng.choice(te, 40_000, replace=False))
Xs, Ys, P0 = X[idx], Y[idx], po[idx]


def r2_ensamble(Xm):
    P = np.clip(np.mean([r.predecir_delta(Xm) for r in redes], 0) + P0[:, None], 0, 6)
    return np.array([r2_score(Ys[:, j], P[:, j]) for j in range(len(hs))])


base = r2_ensamble(Xs)
fam = [familia_de(c) for c in cols]
filas = []
for nombre in FAMILIAS:
    ci = [i for i, x in enumerate(fam) if x == nombre]
    perdidas = []
    for _ in range(5):
        Xp = Xs.copy()
        Xp[:, ci] = Xs[rng.permutation(len(Xs))][:, ci]
        perdidas.append(base - r2_ensamble(Xp))
    perdidas = np.array(perdidas)
    for j, h in enumerate(hs):
        filas.append({"horizonte_min": h, "familia": nombre, "n_variables": len(ci),
                      "perdida_r2": perdidas[:, j].mean(), "perdida_r2_de": perdidas[:, j].std()})
    print(nombre, perdidas.mean(0).round(3), flush=True)
pd.DataFrame(filas).to_csv(TABLAS / "08_importancia_familias.csv", index=False)
