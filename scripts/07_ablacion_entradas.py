"""Paso 7. Ablación de la información de entrada (sección 4.8, hipótesis 3).

Misma MLP entrenada con tres conjuntos de entrada:
  1. solo condiciones actuales (16 variables);
  2. actuales + reloj de fase del ciclo (26 variables);
  3. actuales + reloj de fase + historia del proceso (153 variables).
"""

import numpy as np
import pandas as pd

from po4_prediccion.datos import (ACTUALES, HORIZONTES, OBJETIVO, RELOJ, SEMILLA, TABLAS, asegurar_directorios,
                                  columnas_entrada, construir_dataset, particion_cronologica)
from po4_prediccion.metricas import metricas
from po4_prediccion.red_neuronal import entrenar_mlp

asegurar_directorios()
crudo, f = construir_dataset()
po = crudo[OBJETIVO].to_numpy()
hs = list(HORIZONTES)
Y = np.column_stack([f[f"target_{h}"].to_numpy() for h in hs])
D = Y - po[:, None]
tr, va, te = particion_cronologica(len(f), max(HORIZONTES.values()))
CONFIGURACIONES = {
    "Solo condiciones actuales": ACTUALES,
    "Actuales + reloj de fase": ACTUALES + RELOJ,
    "Actuales + reloj de fase + historia": columnas_entrada(f),
}
filas = []
for nombre, cols in CONFIGURACIONES.items():
    X = f[cols].to_numpy("float32")
    P = entrenar_mlp(X, D, tr, va, semilla=SEMILLA).predecir_delta(X[te]) + po[te, None]
    for j, h in enumerate(hs):
        filas.append({"configuracion": nombre, "n_variables": len(cols), "horizonte_min": h,
                      **metricas(Y[te, j], P[:, j], po[te])})
    print(nombre, "listo", flush=True)
res = pd.DataFrame(filas)
res.to_csv(TABLAS / "07_ablacion_entradas.csv", index=False)
print(res.pivot(index="configuracion", columns="horizonte_min", values="r2").round(3).to_string())
