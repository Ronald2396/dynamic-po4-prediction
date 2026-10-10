"""Paso 3. Modelos de referencia (sección 4.3).

Persistencia, naive de ciclo previo, modelo mecanístico reducido y LightGBM,
evaluados en el mismo conjunto de prueba que la red neuronal.
"""

import time

import numpy as np
import pandas as pd

from po4_prediccion.datos import (HORIZONTES, OBJETIVO, PREDICCIONES, SEMILLA, TABLAS, asegurar_directorios,
                                  columnas_entrada, construir_dataset, particion_cronologica)
from po4_prediccion.referencias import (PARAMETROS_MECANISTICO, ajustar_mecanistico, calendario_fases,
                                        lightgbm_delta, naive_ciclo_previo, persistencia, simular_mecanistico)

asegurar_directorios()
t0 = time.time()
crudo, f = construir_dataset()
X = f[columnas_entrada(f)].to_numpy("float32")
po = persistencia(crudo)
S_media = f["po4_mean_90"].to_numpy()  # media móvil de 3 h
_, _, te = particion_cronologica(len(f), max(HORIZONTES.values()))
rng = np.random.default_rng(SEMILLA)
pred, params = {}, []
for h, pasos in HORIZONTES.items():
    y = f[f"target_{h}"].to_numpy()
    tr, va, te_h = particion_cronologica(len(f), pasos)
    assert (te_h == te).all()
    pred[f"{h}|Persistencia"] = po[te]
    pred[f"{h}|Naive ciclo previo"] = naive_ciclo_previo(crudo, f, pasos)[te]
    cal = calendario_fases(f, pasos)
    sub = np.sort(rng.choice(tr, 150_000, replace=False))
    p = ajustar_mecanistico(po, S_media, cal, y, sub)
    params.append({"horizonte_min": h, **dict(zip(PARAMETROS_MECANISTICO, p))})
    pred[f"{h}|Mecanístico reducido"] = simular_mecanistico(p, po[te], S_media[te], cal[te])
    pred[f"{h}|LightGBM"] = np.clip(lightgbm_delta(X, y, po, tr, va, te), 0, 6)
    print(h, "min listo", round((time.time() - t0) / 60, 1), "min", flush=True)

np.savez_compressed(PREDICCIONES / "referencias.npz", **pred)
pd.DataFrame(params).to_csv(TABLAS / "03_parametros_mecanistico.csv", index=False)
print(pd.DataFrame(params).round(4).to_string())
