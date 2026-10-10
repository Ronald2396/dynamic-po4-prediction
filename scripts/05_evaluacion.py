"""Paso 5. Evaluación en el conjunto de prueba (secciones 4.5–4.7 y 4.9).

Reúne las predicciones de la red neuronal, de los modelos de referencia y de
las redes secuenciales GRU/TCN (notebook de Colab) y calcula:
  * métricas por modelo y horizonte (R², r, RMSE, MAE, sesgo, SS);
  * IC 95 % del R² por bootstrap de bloques diarios;
  * diferencias de R² y prueba de Diebold-Mariano frente a la red propuesta;
  * desempeño de alerta temprana para PO4 > 2 mg/L;
  * R² mensual y RMSE por posición en el ciclo.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, r2_score, recall_score

from po4_prediccion.datos import (COLAB, HORIZONTES, PREDICCIONES, SEMILLA, TABLAS, asegurar_directorios,
                                  construir_dataset, particion_cronologica)
from po4_prediccion.metricas import bootstrap_r2, diebold_mariano, metricas

asegurar_directorios()
hs = list(HORIZONTES)
red = np.load(PREDICCIONES / "red_neuronal.npz")
ref = np.load(PREDICCIONES / "referencias.npz")
Y = red["y"]
P = {k: ref[k] for k in ref.files}
P.update({k: red[k] for k in red.files if k != "y"})
colab_path = COLAB / "predicciones_gru_tcn.npz"
if colab_path.exists():
    col = np.load(colab_path)
    for h in hs:
        P[f"{h}|GRU"] = col[f"{h}|GRU+ctx traj"].astype("float32")
        P[f"{h}|TCN"] = col[f"{h}|TCN+ctx traj"].astype("float32")
MODELOS = [m for m in ["Persistencia", "Naive ciclo previo", "Mecanístico reducido", "LightGBM", "GRU", "TCN",
                       "MLP individual (semilla 42)", "MLP propuesta"] if f"10|{m}" in P]
PROP = "MLP propuesta"

# Métricas
filas = []
for h in hs:
    j = hs.index(h)
    for m in MODELOS:
        filas.append({"horizonte_min": h, "modelo": m, **metricas(Y[:, j], P[f"{h}|{m}"], P[f"{h}|Persistencia"])})
tabla = pd.DataFrame(filas)

# IC bootstrap y pruebas por pares
rng = np.random.default_rng(SEMILLA)
ic, pares = [], []
for h in (10, 30):
    j = hs.index(h)
    bs = bootstrap_r2(Y[:, j], {m: P[f"{h}|{m}"] for m in MODELOS}, rng)
    for m in MODELOS:
        ic.append({"horizonte_min": h, "modelo": m, "r2_ic_inf": np.quantile(bs[m], .025),
                   "r2_ic_sup": np.quantile(bs[m], .975)})
    e_prop = P[f"{h}|{PROP}"] - Y[:, j]
    for m in MODELOS:
        if m == PROP:
            continue
        d = bs[PROP] - bs[m]
        est, pv = diebold_mariano(e_prop, P[f"{h}|{m}"] - Y[:, j], HORIZONTES[h])
        pares.append({"horizonte_min": h, "comparacion": f"{PROP} − {m}", "delta_r2": d.mean(),
                      "ic_inf": np.quantile(d, .025), "ic_sup": np.quantile(d, .975), "dm": est, "p_valor": pv})
tabla = tabla.merge(pd.DataFrame(ic), on=["horizonte_min", "modelo"], how="left")
tabla.to_csv(TABLAS / "05_metricas_prueba.csv", index=False)
pd.DataFrame(pares).to_csv(TABLAS / "05_comparaciones_pareadas.csv", index=False)

# Alerta temprana PO4 > 2 mg/L
alerta = []
for h in (10, 30):
    j = hs.index(h)
    yt = Y[:, j] > 2.0
    for m in ["Persistencia", "LightGBM", "GRU", "TCN", PROP]:
        if f"{h}|{m}" in P:
            yp = P[f"{h}|{m}"] > 2.0
            alerta.append({"horizonte_min": h, "modelo": m, "prevalencia": yt.mean(),
                           "precision": precision_score(yt, yp), "exhaustividad": recall_score(yt, yp),
                           "f1": f1_score(yt, yp)})
pd.DataFrame(alerta).to_csv(TABLAS / "05_alerta_temprana.csv", index=False)

# Desempeño mensual y por posición del ciclo
crudo, f = construir_dataset()
_, _, te = particion_cronologica(len(f), max(HORIZONTES.values()))
mes = crudo["date"].iloc[te].dt.strftime("%Y-%m").to_numpy()
pos = np.minimum((f["cycle_frac"].to_numpy()[te] * 10).astype(int), 12)
mensual, porfase = [], []
for h in (10, 30):
    j = hs.index(h)
    for m_ in sorted(set(mes)):
        k = mes == m_
        mensual.append({"horizonte_min": h, "mes": m_, "r2": r2_score(Y[k, j], P[f"{h}|{PROP}"][k])})
    for m in ("Persistencia", "Mecanístico reducido", PROP):
        e2 = (P[f"{h}|{m}"] - Y[:, j]) ** 2
        for g, v in pd.Series(e2).groupby(pos).mean().items():
            porfase.append({"horizonte_min": h, "modelo": m, "posicion_ciclo": g / 10 + .05, "rmse": v ** .5})
pd.DataFrame(mensual).to_csv(TABLAS / "05_r2_mensual.csv", index=False)
pd.DataFrame(porfase).to_csv(TABLAS / "05_rmse_por_fase.csv", index=False)

pd.set_option("display.width", 200)
print(tabla.round(3).to_string())
print(pd.DataFrame(pares).round(4).to_string())
print(pd.DataFrame(alerta).round(3).to_string())
