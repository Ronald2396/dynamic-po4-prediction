"""Paso 1. Caracterización de los datos (sección 4.1 de la tesis).

Calidad del registro, estadísticos de PO4 por conjunto, ciclo de operación
alternada (duración, varianza intra-ciclo), retención del sensor y
autocorrelación de PO4.
"""

import numpy as np
import pandas as pd

from po4_prediccion.datos import (OBJETIVO, TABLAS, asegurar_directorios, cargar_datos, construir_dataset,
                                  particion_cronologica, reloj_de_fase)
from po4_prediccion.referencias import calendario_fases

asegurar_directorios()
df = cargar_datos()
dt = df["date"].diff().dropna().dt.total_seconds()

# Calidad del registro
calidad = {
    "registros": len(df), "inicio": str(df.date.min()), "fin": str(df.date.max()),
    "frecuencia_s": float(dt.mode().iloc[0]), "huecos_temporales": int((dt != dt.mode().iloc[0]).sum()),
    "faltantes": int(df.isna().sum().sum()), "fechas_duplicadas": int(df.date.duplicated().sum()),
}
pd.Series(calidad).to_csv(TABLAS / "01_calidad_registro.csv", header=["valor"])

# Estadísticos de PO4 por conjunto (partición del horizonte de 30 min)
crudo, f = construir_dataset()
tr, va, te = particion_cronologica(len(f), 15)
filas = []
for nombre, idx in (("Completo", np.arange(len(df))), ("Entrenamiento", tr), ("Validación", va), ("Prueba", te)):
    serie = df[OBJETIVO] if nombre == "Completo" else crudo[OBJETIVO].iloc[idx]
    fechas = df.date if nombre == "Completo" else crudo.date.iloc[idx]
    filas.append({"conjunto": nombre, "desde": fechas.min(), "hasta": fechas.max(), "n": len(serie),
                  "media": serie.mean(), "mediana": serie.median(), "p95": serie.quantile(.95),
                  "fraccion_menor_0_1": (serie < 0.1).mean()})
pd.DataFrame(filas).to_csv(TABLAS / "01_estadisticos_po4.csv", index=False)

# Ciclo de operación alternada
pc = reloj_de_fase(df)
ciclo = (pc["inl"].eq(2) & pc["inl"].shift().eq(1)).cumsum()
rel = df[OBJETIVO] - df[OBJETIVO].groupby(ciclo).transform("mean")
dur = ciclo.map(ciclo.value_counts()) * 2  # minutos
mensual = dur.groupby(df.date.dt.to_period("M")).median()
ciclo_info = {
    "varianza_intraciclo_fraccion": rel.var() / df[OBJETIVO].var(),
    "duracion_mediana_mensual_min_min": mensual.min(), "duracion_mediana_mensual_max_min": mensual.max(),
    "sensor_valores_repetidos_fraccion": float(1 - df[OBJETIVO].ne(df[OBJETIVO].shift()).mean()),
}
# Exactitud del calendario de fases proyectado a +30 min (usado por el modelo mecanístico)
inl = (f["inl"].to_numpy() == 2).astype(int)
real = np.roll(inl, -15)
ciclo_info["calendario_acierto_30min"] = float((calendario_fases(f, 15)[:, -1] == real)[:-15].mean())
ciclo_info["fase_constante_acierto_30min"] = float((inl == real)[:-15].mean())
pd.Series(ciclo_info).to_csv(TABLAS / "01_ciclo_operacional.csv", header=["valor"])

# Autocorrelación de PO4
acf = pd.DataFrame({"desfase_min": np.arange(0, 241, 2)})
acf["acf"] = [1.0] + [df[OBJETIVO].autocorr(int(m // 2)) for m in acf.desfase_min[1:]]
acf.to_csv(TABLAS / "01_autocorrelacion.csv", index=False)

print(pd.Series(calidad).to_string())
print(pd.DataFrame(filas).round(3).to_string())
print(pd.Series(ciclo_info).round(3).to_string())
print(acf[acf.desfase_min.isin([10, 30, 50, 60, 120])].round(3).to_string())
