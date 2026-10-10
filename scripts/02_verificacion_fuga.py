"""Paso 2. Verificación de ausencia de fuga de información (sección 4.2).

Las variables calculadas con la serie truncada en varios puntos de corte
deben coincidir exactamente con las calculadas con la serie completa para
todas las filas anteriores al corte. Si alguna variable usara información
futura, diferiría cerca del corte.
"""

import numpy as np
import pandas as pd

from po4_prediccion.datos import TABLAS, asegurar_directorios, cargar_datos, construir_variables

asegurar_directorios()
crudo = cargar_datos()
completo = construir_variables(crudo)
filas = []
for corte in (100_000, 250_123, 400_007, 500_001):
    parcial = construir_variables(crudo.iloc[:corte].copy())
    a, b = completo.iloc[:corte].to_numpy(float), parcial.to_numpy(float)
    distintas = completo.columns[(~np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9)).any(0)].tolist()
    filas.append({"corte": corte, "n_variables": completo.shape[1], "variables_distintas": len(distintas)})
    print(f"corte {corte}: {len(distintas)} variables distintas de {completo.shape[1]}")
pd.DataFrame(filas).to_csv(TABLAS / "02_verificacion_fuga.csv", index=False)
