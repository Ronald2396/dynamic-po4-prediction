"""Carga de datos, variables de entrada causales y partición cronológica.

Toda variable calculada en el instante de origen t usa únicamente
observaciones con tiempo <= t. El objetivo es T1_PO4(t + h), con h = 10, 30
y 60 min (5, 15 y 30 pasos de 2 min).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data" / "dataset.csv"
RESULTADOS = RAIZ / "resultados"
TABLAS = RESULTADOS / "tablas"
PREDICCIONES = RESULTADOS / "predicciones"
MODELOS = RESULTADOS / "modelos"
COLAB = RESULTADOS / "colab"
FIGURAS = RAIZ / "figuras"

OBJETIVO = "T1_PO4"
HORIZONTES = {10: 5, 30: 15, 60: 30}  # minutos -> pasos de 2 min
CALENTAMIENTO = 400  # pasos iniciales descartados para que toda ventana esté definida
SEMILLA = 42


def asegurar_directorios() -> None:
    for d in (TABLAS, PREDICCIONES, MODELOS, COLAB, FIGURAS):
        d.mkdir(parents=True, exist_ok=True)


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV SCADA de la PTAR de Agtrup y lo ordena cronológicamente."""
    df = pd.read_csv(DATOS, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Reloj de fase de la operación alternada
# --------------------------------------------------------------------------
def _pasos_desde_cambio(s: pd.Series) -> np.ndarray:
    """Número de pasos transcurridos desde el último cambio de valor de s."""
    grupo = s.ne(s.shift()).cumsum()
    return s.groupby(grupo).cumcount().to_numpy()


def _duracion_tramo_previo(s: pd.Series, k: int = 1) -> np.ndarray:
    """Duración del k-ésimo tramo *completo* anterior de s (causal)."""
    grupo = s.ne(s.shift()).cumsum().to_numpy()
    duraciones = pd.Series(grupo).value_counts().sort_index().to_numpy()
    previo = np.full(len(grupo), np.nan)
    idx = grupo - 1 - k
    ok = idx >= 0
    previo[ok] = duraciones[idx[ok]]
    return previo


def reloj_de_fase(d: pd.DataFrame) -> pd.DataFrame:
    """Variables que ubican el instante t dentro del ciclo de operación alternada.

    Un ciclo comienza cuando la fase de entrada pasa de 1 a 2 (inicio de la
    alimentación del tanque 1).
    """
    inl = d["PROCESSPHASE_INLET"].round().clip(1, 2).astype(int)
    out = d["PROCESSPHASE_OUTLET"].round().clip(1, 2).astype(int)
    f = pd.DataFrame(index=d.index)
    f["inl"] = inl
    f["out"] = out
    f["inl_out_state"] = (inl - 1) * 2 + (out - 1)
    f["since_inl_change"] = _pasos_desde_cambio(inl)
    f["since_out_change"] = _pasos_desde_cambio(out)
    f["prev_inl_run"] = _duracion_tramo_previo(inl, 1)
    f["prev_inl_run2"] = _duracion_tramo_previo(inl, 2)
    f["prev_out_run"] = _duracion_tramo_previo(out, 1)
    inicio = inl.eq(2) & inl.shift().eq(1)
    ciclo = inicio.cumsum()
    f["cycle_pos"] = d.groupby(ciclo).cumcount().to_numpy()
    # Duración del último ciclo completo, conocida en el instante t.
    ciclo_np = ciclo.to_numpy()
    tamanos = ciclo.value_counts().sort_index()
    ultimo = np.full(len(d), np.nan)
    previo = ciclo_np - 1
    m = previo >= tamanos.index.min()
    ultimo[m] = tamanos.reindex(previo[m]).to_numpy()
    f["last_cycle_len"] = ultimo
    f["cycle_frac"] = f["cycle_pos"] / f["last_cycle_len"]
    # Reloj de aireación: O2 > 0.3 mg/L se considera aireado.
    aire = (d["T1_O2"] > 0.3).astype(int)
    f["aerated"] = aire
    f["since_aer_change"] = _pasos_desde_cambio(aire)
    return f


# --------------------------------------------------------------------------
# Conjunto completo de 153 variables de entrada
# --------------------------------------------------------------------------
def construir_variables(d: pd.DataFrame) -> pd.DataFrame:
    po = d[OBJETIVO]
    f = reloj_de_fase(d)
    for c in ["IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q", "MAX_CF", "T1_NH4", "T1_PO4"]:
        f[c] = d[c]
    # Historia densa de PO4 (aprox. dos ciclos).
    for lag in list(range(1, 11)) + list(range(12, 31, 2)) + list(range(33, 91, 3)):
        f[f"po4_l{lag}"] = po.shift(lag)
    for lag in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60):
        f[f"nh4_l{lag}"] = d["T1_NH4"].shift(lag)
        f[f"o2_l{lag}"] = d["T1_O2"].shift(lag)
    for lag in (1, 2, 5, 10, 20):
        f[f"inl_l{lag}"] = f["inl"].shift(lag)
        f[f"q_l{lag}"] = d["IN_Q"].shift(lag)
    # Estadísticos móviles (incluyen t; causales).
    for w in (5, 15, 40, 90, 360, 720):
        f[f"po4_mean_{w}"] = po.rolling(w).mean()
        f[f"po4_max_{w}"] = po.rolling(w).max()
        f[f"po4_min_{w}"] = po.rolling(w).min()
        f[f"nh4_mean_{w}"] = d["T1_NH4"].rolling(w).mean()
        f[f"o2_mean_{w}"] = d["T1_O2"].rolling(w).mean()
        f[f"q_mean_{w}"] = d["IN_Q"].rolling(w).mean()
        f[f"fe_sum_{w}"] = d["IN_METAL_Q"].rolling(w).sum()
        f[f"pac_sum_{w}"] = d["METAL_Q"].rolling(w).sum()
    f["po4_std_40"] = po.rolling(40).std()
    f["po4_d1"] = po.diff(1)
    f["po4_d3"] = po.diff(3)
    f["nh4_d3"] = d["T1_NH4"].diff(3)
    f["since_po4_change"] = _pasos_desde_cambio(po)
    # Valor en la misma posición del ciclo anterior (t - duración del último ciclo).
    L = f["last_cycle_len"].fillna(40).clip(5, 300).astype(int).to_numpy()
    idx = np.clip(np.arange(len(d)) - L, 0, None)
    f["po4_prev_cycle_same_pos"] = po.to_numpy()[idx]
    f["nh4_prev_cycle_same_pos"] = d["T1_NH4"].to_numpy()[idx]
    minuto = d["date"].dt.hour * 60 + d["date"].dt.minute
    dia = d["date"].dt.dayofyear
    f["tod_sin"] = np.sin(2 * np.pi * minuto / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * minuto / 1440)
    f["dow"] = d["date"].dt.dayofweek
    f["year_sin"] = np.sin(2 * np.pi * dia / 365.25)
    f["year_cos"] = np.cos(2 * np.pi * dia / 365.25)
    return f


FAMILIAS = {
    "PO₄ actual": lambda c: c == "T1_PO4",
    "Historia de PO₄": lambda c: c.startswith("po4_") or c == "since_po4_change",
    "Fase y reloj del ciclo": lambda c: c in {"inl", "out", "inl_out_state", "since_inl_change", "since_out_change",
                                              "prev_inl_run", "prev_inl_run2", "prev_out_run", "cycle_pos",
                                              "last_cycle_len", "cycle_frac", "aerated", "since_aer_change"}
                                    or c.startswith("inl_l"),
    "NH₄ (actual e historia)": lambda c: c == "T1_NH4" or c.startswith("nh4_"),
    "O₂ disuelto": lambda c: c == "T1_O2" or c.startswith("o2_"),
    "Caudal y carga hidráulica": lambda c: c in {"IN_Q", "MAX_CF"} or c.startswith("q_"),
    "Dosificación química (Fe, PAC)": lambda c: c in {"IN_METAL_Q", "METAL_Q"} or c.startswith(("fe_", "pac_")),
    "Temperatura y calendario": lambda c: c in {"TEMPERATURE", "tod_sin", "tod_cos", "dow", "year_sin", "year_cos"},
}

ACTUALES = ["inl", "out", "inl_out_state", "IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q",
            "MAX_CF", "T1_NH4", "T1_PO4", "tod_sin", "tod_cos", "dow", "year_sin", "year_cos"]
RELOJ = ["since_inl_change", "since_out_change", "prev_inl_run", "prev_inl_run2", "prev_out_run",
         "cycle_pos", "last_cycle_len", "cycle_frac", "aerated", "since_aer_change"]


def familia_de(columna: str) -> str:
    hits = [k for k, fn in FAMILIAS.items() if fn(columna)]
    assert len(hits) == 1, (columna, hits)
    return hits[0]


def construir_dataset():
    """Devuelve (datos crudos, variables + objetivos) alineados por origen."""
    crudo = cargar_datos()
    f = construir_variables(crudo)
    for minutos, pasos in HORIZONTES.items():
        f[f"target_{minutos}"] = crudo[OBJETIVO].shift(-pasos)
    keep = slice(CALENTAMIENTO, len(crudo) - max(HORIZONTES.values()))
    return crudo.iloc[keep].reset_index(drop=True), f.iloc[keep].reset_index(drop=True)


def columnas_entrada(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if not c.startswith("target_")]


def particion_cronologica(n: int, pasos: int):
    """70/15/15 cronológico; se purgan `pasos` orígenes antes de cada frontera
    para que ninguna etiqueta futura cruce al conjunto siguiente."""
    n0, n1 = int(n * .70), int(n * .85)
    return np.arange(0, n0 - pasos), np.arange(n0, n1 - pasos), np.arange(n1, n)


FOLDS_ESTACIONALES = [("2022-08-01", "2022-11-01"), ("2022-11-01", "2023-02-01"),
                      ("2023-02-01", "2023-05-01"), ("2023-05-01", "2023-08-01")]
DIAS_VALIDACION = 30


def particion_fold(fechas: pd.Series, inicio: str, fin: str, pasos: int):
    """Validación por origen móvil: entrenamiento = todo hasta 30 días antes
    del trimestre; validación = esos 30 días; prueba = el trimestre."""
    s, e = pd.Timestamp(inicio, tz="UTC"), pd.Timestamp(fin, tz="UTC")
    v = s - pd.Timedelta(days=DIAS_VALIDACION)
    d = fechas.to_numpy()
    n_v, n_s, n_e = np.searchsorted(d, v), np.searchsorted(d, s), np.searchsorted(d, e)
    return np.arange(0, n_v - pasos), np.arange(n_v, n_s - pasos), np.arange(n_s, n_e - pasos)
