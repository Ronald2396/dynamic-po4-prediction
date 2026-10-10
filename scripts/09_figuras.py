"""Paso 9. Figuras de la tesis (PNG, 300 dpi) en figuras/.

Las figuras no llevan título interno: el título va en el pie de figura.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, r2_score

from po4_prediccion.datos import (CALENTAMIENTO, COLAB, FIGURAS, HORIZONTES, OBJETIVO, PREDICCIONES, TABLAS,
                                  asegurar_directorios, cargar_datos, construir_dataset, particion_cronologica,
                                  reloj_de_fase)

asegurar_directorios()
AZUL, NARANJA, AQUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GRIS, GRIS2, GRIS3 = "#8a8984", "#b5b4ae", "#5f5e5a"
TINTA, TINTA2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "legend.frameon": False, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLOR = {"Persistencia": GRIS2, "Naive ciclo previo": GRIS, "Mecanístico reducido": GRIS3,
         "LightGBM": AZUL, "GRU": AQUA, "TCN": VIOLETA, "MLP propuesta": NARANJA}
ORDEN = ["Persistencia", "Naive ciclo previo", "Mecanístico reducido", "LightGBM", "GRU", "TCN", "MLP propuesta"]
METAS = {10: 0.70, 30: 0.60}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
COLOR_CONJ = {"Entrenamiento": "#f1f0ec", "Validación": "#e3eefb", "Prueba": "#fde9e1"}


def guardar(fig, nombre):
    fig.savefig(FIGURAS / nombre); plt.close(fig); print("guardada", nombre)


def etiqueta(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom", ha="left")


def fmt_mes(dos_lineas=False):
    sep = "\n" if dos_lineas else " "
    return mticker.FuncFormatter(lambda x, _: (lambda d: f"{MESES[d.month - 1]}{sep}{d.year}")(mdates.num2date(x)))


def fmt_fecha_hora():
    return mticker.FuncFormatter(lambda x, _: (lambda d: f"{d.day}-{MESES[d.month - 1]}\n{d:%H:%M}")(mdates.num2date(x)))


# ---------------------------------------------------------------- datos
completo = cargar_datos()
crudo, f = construir_dataset()
hs = list(HORIZONTES)
n = len(f)
_, _, te = particion_cronologica(n, max(HORIZONTES.values()))
n0, n1 = int(n * .7), int(n * .85)
cortes = [crudo.date.iloc[0], crudo.date.iloc[n0], crudo.date.iloc[n1], crudo.date.iloc[-1]]
red = np.load(PREDICCIONES / "red_neuronal.npz")
ref = np.load(PREDICCIONES / "referencias.npz")
P = {k: ref[k] for k in ref.files}
P.update({k: red[k] for k in red.files})
Y = P.pop("y")
colab = np.load(COLAB / "predicciones_gru_tcn.npz")
for h in hs:
    P[f"{h}|GRU"] = colab[f"{h}|GRU+ctx traj"].astype("float32")
    P[f"{h}|TCN"] = colab[f"{h}|TCN+ctx traj"].astype("float32")
met = pd.read_csv(TABLAS / "05_metricas_prueba.csv")
fechas_te = crudo.date.to_numpy()[te]


def sombrear(ax, textos=True):
    for (a, b), (nombre, c) in zip(zip(cortes[:-1], cortes[1:]), COLOR_CONJ.items()):
        ax.axvspan(a, b, color=c, lw=0, zorder=0)
        if textos:
            ax.text(a + (b - a) / 2, .97, nombre, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=TINTA2, fontsize=8.5)


# ---------------------------------------------------------------- F01 diagrama metodológico
fig, ax = plt.subplots(figsize=(9, 12))
ax.set_xlim(0, 10); ax.set_ylim(-0.75, 13.6); ax.axis("off")


def caja(x, y, w, h, titulo, texto, color, script=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="#9b9a94", lw=1))
    ax.text(x + w / 2, y + h - .2, titulo, ha="center", va="top", fontweight="bold", fontsize=9.5, color=TINTA)
    ax.text(x + w / 2, y + h - .52, texto, ha="center", va="top", fontsize=8.2, color=TINTA2, linespacing=1.35)
    if script:
        ax.text(x + w / 2, y + .1, script, ha="center", va="bottom", fontsize=7.2, color=GRIS3, family="monospace")


def flecha(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=GRIS3, lw=1.2))


C1, C2, C3 = "#eef4fc", "#fdf0ea", "#eaf7f1"
caja(1.5, 12.4, 7, 1.15, "1. Datos SCADA de la PTAR de Agtrup", "525 600 registros cada 2 min (ago-2021 a jul-2023)\n10 señales: PO₄, NH₄, O₂, caudal, Fe, PAC, temperatura, fases", C1)
caja(1.5, 10.45, 7, 1.45, "2. Control de calidad y caracterización", "Faltantes, duplicados, huecos · ciclo de operación alternada\nvarianza intra-ciclo · autocorrelación", C1, "scripts/01_caracterizacion_datos.py")
caja(1.5, 8.5, 7, 1.45, "3. Ingeniería de variables causales", "153 variables en 8 familias: reloj de fase del ciclo,\nhistoria de PO₄/NH₄/O₂, caudal, dosificación, calendario", C1, "src/po4_prediccion/datos.py")
caja(1.5, 6.55, 7, 1.45, "4. Verificación anti-fuga y partición cronológica", "Recalculo con la serie truncada (0 variables distintas)\n70 % entrenamiento · 15 % validación · 15 % prueba · purga de 60 min", C1, "scripts/02_verificacion_fuga.py")
caja(0.1, 3.95, 3.1, 2.1, "5a. Modelos de referencia", "Persistencia\nNaive de ciclo previo\nMecanístico reducido\nLightGBM", C3, "03_modelos_referencia.py")
caja(3.45, 3.95, 3.1, 2.1, "5b. Red neuronal propuesta", "MLP 3 × 512 (GELU, LayerNorm)\nsalida ΔPO₄ a 10/30/60 min\nAdamW + Huber + det. temprana\nensamble de 5 semillas", C2, "04_red_neuronal.py")
caja(6.8, 3.95, 3.1, 2.1, "5c. Redes secuenciales", "GRU + contexto\nTCN + contexto\nventanas de 6 h (GPU)", C3, "notebook de Colab")
caja(1.5, 1.6, 7, 1.85, "6. Evaluación en el conjunto de prueba", "R², r de Pearson, RMSE, MAE, sesgo, índice de habilidad SS\nIC 95 % por bootstrap de bloques diarios · prueba de Diebold–Mariano\nalerta temprana (precisión, exhaustividad, F1)", C1, "scripts/05_evaluacion.py")
caja(0.1, -0.7, 3.1, 1.8, "7a. Robustez", "Validación por origen\nmóvil (4 trimestres)\n5 semillas", C1, "06_backtest_estacional.py")
caja(3.45, -0.7, 3.1, 1.8, "7b. Información de entrada", "Ablación: actuales →\n+ reloj de fase → + historia", C1, "07_ablacion_entradas.py")
caja(6.8, -0.7, 3.1, 1.8, "7c. Explicabilidad", "Importancia por\npermutación agrupada\npor familias", C1, "08_explicabilidad.py")
for y_sup, y_inf in ((12.4, 11.9), (10.45, 9.95), (8.5, 8.0)):
    flecha(5, y_sup, 5, y_inf + .02)
for x in (1.65, 5.0, 8.35):
    flecha(5, 6.55, x, 6.07)
    flecha(x, 3.95, 5 if x == 5.0 else (3.0 if x < 5 else 7.0), 3.47)
    flecha(5 if x == 5.0 else (3.0 if x < 5 else 7.0), 1.6, x, 1.12)
guardar(fig, "F01_diagrama_metodologia.png")

# ---------------------------------------------------------------- F01b diagrama del protocolo de validación
fig, ax = plt.subplots(figsize=(10, 7.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 9.6); ax.axis("off")
# Línea de tiempo de la partición
ax.text(5, 9.35, "Partición cronológica (sin mezcla aleatoria)", ha="center", fontweight="bold", fontsize=10)
for x0, w_, nombre, c in ((0.3, 6.3, "Entrenamiento 70 %\nago-2021 – dic-2022", "#f1f0ec"),
                          (6.65, 1.35, "Validación 15 %\ndic-22 – abr-23", "#e3eefb"),
                          (8.05, 1.65, "Prueba 15 %\nabr-23 – jul-23", "#fde9e1")):
    ax.add_patch(FancyBboxPatch((x0, 8.15), w_, .85, boxstyle="round,pad=0.01,rounding_size=0.05", fc=c, ec="#9b9a94"))
    ax.text(x0 + w_ / 2, 8.57, nombre, ha="center", va="center", fontsize=8.2)
ax.text(5, 7.85, "purga de 60 min en cada frontera · normalización y detención temprana solo con entrenamiento/validación",
        ha="center", fontsize=7.8, color=TINTA2)
caja(0.3, 6.0, 3.0, 1.45, "Ajuste", "Pesos de la red con\nentrenamiento (AdamW,\npérdida de Huber)", C2)
caja(3.5, 6.0, 3.0, 1.45, "Detención temprana", "MSE de validación;\nse conserva la mejor\népoca (paciencia 6)", C2)
caja(6.7, 6.0, 3.0, 1.45, "Predicción en prueba", "78 776 orígenes no vistos\na 10, 30 y 60 min\n(ensamble de 5 semillas)", C2)
flecha(3.3, 6.72, 3.5, 6.72); flecha(6.5, 6.72, 6.7, 6.72)
caja(0.3, 3.55, 4.6, 1.95, "Métricas puntuales", "R²  ·  r de Pearson  ·  RMSE  ·  MAE\nsesgo  ·  índice de habilidad\nSS = 1 − MSE / MSE_persistencia", C1)
caja(5.1, 3.55, 4.6, 1.95, "Inferencia estadística", "IC 95 % del R²: bootstrap de bloques\ndiarios (1000 remuestreos)\nDiebold–Mariano frente a cada referencia", C1)
flecha(8.2, 6.0, 2.6, 5.5); flecha(8.2, 6.0, 7.4, 5.5)
caja(0.3, 1.0, 3.0, 2.0, "Robustez temporal", "Origen móvil:\n4 trimestres\n(ago-22 a jul-23)", C3)
caja(3.5, 1.0, 3.0, 2.0, "Robustez del\nentrenamiento", "\n5 semillas de\ninicialización", C3)
caja(6.7, 1.0, 3.0, 2.0, "Utilidad operativa", "Alerta PO₄ > 2 mg/L:\nprecisión, exhaustividad,\nF1", C3)
for x in (1.8, 5.0, 8.2):
    flecha(5, 3.55, x, 3.0)
ax.text(5, .45, "Criterio de éxito: R² > 0.70 a 10 min y R² > 0.60 a 30 min (objetivo general)",
        ha="center", fontsize=9, fontweight="bold", color=NARANJA)
guardar(fig, "F01b_diagrama_validacion.png")

# ---------------------------------------------------------------- F02 serie PO4
dia = completo.set_index("date")[OBJETIVO].resample("1D")
dm, d05, d95 = dia.mean(), dia.quantile(.05), dia.quantile(.95)
fig, ax = plt.subplots(figsize=(9, 3.4))
sombrear(ax)
ax.fill_between(dm.index, d05, d95, color=AZUL, alpha=.18, lw=0, label="Percentiles 5–95 diarios")
ax.plot(dm.index, dm, color=AZUL, lw=1, label="Media diaria")
ax.set(ylabel="PO₄ en tanque 1 (mg/L)", ylim=(0, None))
ax.xaxis.set_major_formatter(fmt_mes())
ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, .9))
guardar(fig, "F02_serie_po4_particion.png")

# ---------------------------------------------------------------- F03 ventana del ciclo
w = completo[(completo.date >= "2022-06-14 04:00") & (completo.date < "2022-06-14 12:00")]
fig, axes = plt.subplots(4, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.3, 1.3, .9], "hspace": .35})
for ax, (col, lab, c) in zip(axes, [("T1_PO4", "PO₄ (mg/L)", AZUL), ("T1_NH4", "NH₄ (mg/L)", AQUA), ("T1_O2", "O₂ (mg/L)", VIOLETA)]):
    ax.plot(w.date, w[col], color=c, lw=1.4, drawstyle="steps-post")
    ax.set_ylabel(lab)
alim = (w.PROCESSPHASE_INLET.round() == 2).astype(int)
axes[3].fill_between(w.date, 0, alim, step="post", color=NARANJA, alpha=.55, lw=0)
axes[3].set(ylabel="Alim. T1", yticks=[0, 1], yticklabels=["no", "sí"], ylim=(0, 1.1))
for ax in axes[:3]:
    for a, b in zip(w.date[alim.diff().eq(1)], w.date[alim.diff().eq(-1)]):
        ax.axvspan(a, b, color=NARANJA, alpha=.08, lw=0)
axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[3].set_xlabel("14 de junio de 2022 (hora)")
for ax, l in zip(axes, "abcd"):
    etiqueta(ax, f"({l})")
fig.align_ylabels(axes)
guardar(fig, "F03_ventana_ciclo_operacional.png")

# ---------------------------------------------------------------- F04 perfiles intra-ciclo
pc = reloj_de_fase(completo)
ciclo = (pc["inl"].eq(2) & pc["inl"].shift().eq(1)).cumsum()
b = ((pc["cycle_pos"] / pc["last_cycle_len"]).clip(0, 1.2) * 24).round() / 24
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, (col, lab, c) in zip(axes, [("T1_PO4", "PO₄ − media del ciclo (mg/L)", AZUL),
                                     ("T1_NH4", "NH₄ − media del ciclo (mg/L)", AQUA), ("T1_O2", "O₂ disuelto (mg/L)", VIOLETA)]):
    s = completo[col]
    v = s if col == "T1_O2" else s - s.groupby(ciclo).transform("mean")
    g = v.groupby(b)
    ax.fill_between(g.mean().index, g.quantile(.25), g.quantile(.75), color=c, alpha=.18, lw=0, label="Rango intercuartil")
    ax.plot(g.mean().index, g.mean(), color=c, lw=2, label="Media")
    if col != "T1_O2":
        ax.axhline(0, color=TINTA2, lw=.7)
    ax.set(xlabel="Posición en el ciclo (fracción)", ylabel=lab)
axes[0].legend(fontsize=8, loc="lower left")
for ax, l in zip(axes, "abc"):
    etiqueta(ax, f"({l})")
fig.tight_layout()
guardar(fig, "F04_perfil_intraciclo.png")

# ---------------------------------------------------------------- F05 autocorrelación
acf = pd.read_csv(TABLAS / "01_autocorrelacion.csv")
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(acf.desfase_min, acf.acf, color=AZUL, lw=2)
for m in (10, 30):
    ax.axvline(m, color=GRIS, ls="--", lw=1)
    ax.text(m + 2, .95, f"{m} min", color=TINTA2, fontsize=8.5)
ax.set(xlabel="Desfase (min)", ylabel="Autocorrelación de PO₄", ylim=(0, 1.02), xlim=(0, 240))
guardar(fig, "F05_autocorrelacion.png")

# ---------------------------------------------------------------- F06 condiciones mensuales
mon = completo.set_index("date").resample("MS")
ms = pd.DataFrame({"PO₄ medio (mg/L)": mon.T1_PO4.mean(), "Temperatura (°C)": mon.TEMPERATURE.mean(),
                   "Caudal medio (m³/h)": mon.IN_Q.mean(), "PAC al decantador (L/h)": mon.METAL_Q.mean()})
fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
for ax, col, c, l in zip(axes.flat, ms.columns, [AZUL, NARANJA, AQUA, VIOLETA], "abcd"):
    sombrear(ax, False)
    ax.plot(ms.index + pd.Timedelta(days=14), ms[col], color=c, lw=1.8, marker="o", ms=3.5)
    ax.set_ylabel(col)
    etiqueta(ax, f"({l})")
for ax in axes[1]:
    ax.xaxis.set_major_formatter(fmt_mes(True))
fig.tight_layout()
guardar(fig, "F06_condiciones_mensuales.png")

# ---------------------------------------------------------------- F07 curvas de entrenamiento
hist = pd.read_csv(TABLAS / "04_curvas_entrenamiento.csv")
fig, ax = plt.subplots(figsize=(7, 3.4))
for s, g in hist.groupby("semilla"):
    ref42 = s == 42
    kw = dict(lw=2 if ref42 else .8, alpha=1 if ref42 else .35)
    ax.plot(g.epoca, g.mse_entrenamiento, color=AZUL, label="Entrenamiento" if ref42 else None, **kw)
    ax.plot(g.epoca, g.mse_validacion, color=NARANJA, label="Validación" if ref42 else None, **kw)
    if ref42:
        mejor = g.loc[g.mse_validacion.idxmin()]
        ax.scatter(mejor.epoca, mejor.mse_validacion, s=60, facecolor=FONDO, edgecolor=NARANJA, lw=2, zorder=5,
                   label="Mejor época (detención temprana)")
ax.set(xlabel="Época", ylabel="MSE (escala estandarizada)")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=8.5)
guardar(fig, "F07_curvas_entrenamiento.png")

# ---------------------------------------------------------------- F08 comparación R²
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, h, l in zip(axes, (10, 30), "ab"):
    mh = met[met.horizonte_min == h].set_index("modelo")
    vals = np.array([mh.loc[m, "r2"] for m in ORDEN])
    err = np.array([[v - mh.loc[m, "r2_ic_inf"], mh.loc[m, "r2_ic_sup"] - v] for m, v in zip(ORDEN, vals)]).T
    y = np.arange(len(ORDEN))
    ax.barh(y, vals, color=[COLOR[m] for m in ORDEN], height=.62, xerr=err, error_kw={"lw": 1.2, "capsize": 3})
    for yy, v, hi in zip(y, vals, err[1]):
        ax.text(max(v + hi, 0) + .03, yy, f"{v:.2f}", va="center", fontsize=8.5)
    ax.axvline(METAS[h], color=TINTA, ls="--", lw=1)
    ax.text(METAS[h], len(ORDEN) - .35, f" meta R² > {METAS[h]:.2f}", fontsize=8)
    ax.axvline(0, color=TINTA2, lw=.8)
    ax.set(xlim=(-0.7, 1.05), xlabel=f"R² en prueba — horizonte {h} min")
    ax.set_yticks(y, ORDEN)
    etiqueta(ax, f"({l})")
fig.tight_layout()
guardar(fig, "F08_comparacion_r2.png")

# ---------------------------------------------------------------- F09 R² vs horizonte
K = 30
serie = completo[OBJETIVO].to_numpy()[CALENTAMIENTO:]
yk = np.stack([serie[k:k + n] for k in range(1, K + 1)], 1)[te]
po_te = crudo[OBJETIVO].to_numpy()[te]
minutos = np.arange(1, K + 1) * 2
fig, ax = plt.subplots(figsize=(7.5, 4))
for nombre, clave, c in [("MLP", "MLP-v2 traj", NARANJA), ("GRU", "GRU+ctx traj", AQUA), ("TCN", "TCN+ctx traj", VIOLETA)]:
    T = colab[f"traj|{clave}"].astype("float32")
    ax.plot(minutos, [r2_score(yk[:, k], np.clip(T[:, k], 0, 6)) for k in range(K)], color=c, lw=2, label=nombre)
ax.plot(minutos, [r2_score(yk[:, k], po_te) for k in range(K)], color=GRIS, lw=2, ls="--", label="Persistencia")
for h in (10, 30):
    ax.plot([h - 3, h + 3], [METAS[h]] * 2, color=TINTA, lw=2)
    ax.text(h, METAS[h] - .05, f"meta {h} min", ha="center", va="top", fontsize=8)
ax.axhline(0, color=TINTA2, lw=.7)
ax.set(xlabel="Horizonte de predicción (min)", ylabel="R² en prueba", ylim=(-1.05, 1), xlim=(0, 62))
ax.legend(loc="lower left", fontsize=8.5)
guardar(fig, "F09_r2_vs_horizonte.png")

# ---------------------------------------------------------------- F10 observado vs predicho
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, h, l in zip(axes, (10, 30), "ab"):
    j = hs.index(h); y = Y[:, j]; p = P[f"{h}|MLP propuesta"]
    hb = ax.hexbin(y, p, gridsize=70, bins="log", cmap="Blues", mincnt=1, extent=(0, 5, 0, 5))
    ax.plot([0, 5], [0, 5], color=NARANJA, lw=1.5, ls="--", label="Línea 1:1")
    ax.set(xlabel="PO₄ observado (mg/L)", ylabel="PO₄ predicho (mg/L)", xlim=(0, 5), ylim=(0, 5), aspect="equal")
    ax.text(.04, .95, f"Horizonte {h} min\nR² = {r2_score(y, p):.3f}", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    etiqueta(ax, f"({l})")
cb = fig.colorbar(hb, ax=axes, shrink=.8, pad=.02); cb.set_label("Número de puntos (escala log)")
guardar(fig, "F10_observado_vs_predicho.png")

# ---------------------------------------------------------------- F11 serie de prueba
s30 = pd.Series(Y[:, hs.index(30)])
ventana = 720
desv = s30.rolling(ventana).std().to_numpy()
cand = np.arange(ventana, len(s30), 60)
fin = cand[np.nanargmax(desv[cand])]; ini = fin - ventana
fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
for ax, h, l in zip(axes, (10, 30), "ab"):
    j = hs.index(h)
    t = pd.to_datetime(fechas_te[ini:fin]) + pd.Timedelta(minutes=h)
    ax.plot(t, Y[ini:fin, j], color=TINTA, lw=1.3, label="Observado")
    ax.plot(t, P[f"{h}|Persistencia"][ini:fin], color=GRIS2, lw=1.2, ls="--", label="Persistencia")
    ax.plot(t, P[f"{h}|MLP propuesta"][ini:fin], color=NARANJA, lw=1.5, label="MLP propuesta")
    ax.set_ylabel(f"PO₄ (mg/L), {h} min")
    etiqueta(ax, f"({l})")
axes[0].legend(ncol=3, fontsize=8.5, loc="upper left")
axes[1].xaxis.set_major_formatter(fmt_fecha_hora())
fig.tight_layout()
guardar(fig, "F11_serie_prueba.png")

# ---------------------------------------------------------------- F12 residuos
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
bins = np.linspace(-2, 2, 81)
for ax, h, l in zip(axes, (10, 30), "ab"):
    j = hs.index(h)
    for nombre, c, a in (("Persistencia", GRIS2, .5), ("MLP propuesta", NARANJA, .65)):
        r = P[f"{h}|{nombre}"] - Y[:, j]
        ax.hist(r, bins=bins, color=c, alpha=a, density=True, label=f"{nombre} (sesgo {r.mean():+.3f})")
    ax.axvline(0, color=TINTA, lw=.8)
    ax.set(xlabel=f"Error de predicción, {h} min (mg/L)", ylabel="Densidad")
    ax.legend(fontsize=8)
    etiqueta(ax, f"({l})")
fig.tight_layout()
guardar(fig, "F12_distribucion_residuos.png")

# ---------------------------------------------------------------- F13 backtest
bt = pd.read_csv(TABLAS / "06_backtest_trimestres.csv")
trims = list(dict.fromkeys(bt.trimestre))
etiq = ["ago–oct\n2022", "nov 2022–\nene 2023", "feb–abr\n2023", "may–jul\n2023"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, h, l in zip(axes, (10, 30), "ab"):
    for i, (m, c) in enumerate([("Persistencia", GRIS2), ("LightGBM", AZUL), ("MLP propuesta", NARANJA)]):
        v = bt[(bt.horizonte_min == h) & (bt.modelo == m)].set_index("trimestre").r2.reindex(trims).values
        x = np.arange(4) + (i - 1) * .27
        ax.bar(x, v, .25, color=c, label="MLP (1 semilla)" if m == "MLP propuesta" else m)
        if m == "MLP propuesta":
            for xx, vv in zip(x, v):
                ax.text(xx, vv + .02, f"{vv:.2f}", ha="center", fontsize=8)
    ax.axhline(METAS[h], color=TINTA, ls="--", lw=1)
    ax.axhline(0, color=TINTA2, lw=.7)
    ax.set_xticks(range(4), etiq, fontsize=8.5)
    ax.set(ylabel=f"R² — {h} min", ylim=(-0.6, 1.05))
    etiqueta(ax, f"({l})")
axes[0].legend(fontsize=8, loc="lower left", ncol=3)
fig.tight_layout()
guardar(fig, "F13_backtest_estacional.png")

# ---------------------------------------------------------------- F14 semillas
sm = pd.read_csv(TABLAS / "04_metricas_por_semilla.csv")
fig, ax = plt.subplots(figsize=(6, 3.4))
for i, h in enumerate(hs):
    v = sm[sm.horizonte_min == h].r2.values
    ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color=NARANJA, s=40, zorder=3)
    ax.plot([i - .25, i + .25], [v.mean()] * 2, color=TINTA, lw=2)
    ax.text(i + .3, v.mean(), f"{v.mean():.3f} ± {v.std(ddof=1):.3f}", va="center", fontsize=8.5)
    if h in METAS:
        ax.plot([i - .35, i + .35], [METAS[h]] * 2, color=TINTA, ls="--", lw=1)
ax.set_xticks(range(len(hs)), [f"{h} min" for h in hs]); ax.set_xlim(-.5, 2.9)
ax.set(ylabel="R² en prueba (MLP individual)", ylim=(0.4, 0.9))
guardar(fig, "F14_robustez_semillas.png")

# ---------------------------------------------------------------- F15 ablación
ab = pd.read_csv(TABLAS / "07_ablacion_entradas.csv")
confs = list(dict.fromkeys(ab.configuracion))
fig, ax = plt.subplots(figsize=(8, 3.2))
for i, (cf, c) in enumerate(zip(confs, [GRIS2, AZUL, NARANJA])):
    v = ab[ab.configuracion == cf].set_index("horizonte_min").r2.reindex(hs).values
    x = np.arange(3) + (i - 1) * .26
    ax.bar(x, v, .23, color=c, label=cf)
    for xx, vv in zip(x, v):
        ax.text(xx, vv + .015, f"{vv:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(3), [f"{h} min" for h in hs]); ax.set(ylabel="R² en prueba", ylim=(0, 1))
ax.legend(fontsize=8.5, loc="upper right")
guardar(fig, "F15_ablacion_entradas.png")

# ---------------------------------------------------------------- F16 importancia
imp = pd.read_csv(TABLAS / "08_importancia_familias.csv")
fams = imp[imp.horizonte_min == 30].sort_values("perdida_r2").familia.tolist()
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(fams))
for i, (h, c) in enumerate([(10, AZUL), (30, NARANJA)]):
    d = imp[imp.horizonte_min == h].set_index("familia").reindex(fams)
    ax.barh(y + (i - .5) * .38, d.perdida_r2, .36, xerr=d.perdida_r2_de, color=c, label=f"{h} min",
            error_kw={"lw": 1, "capsize": 2})
ax.set_yticks(y, fams)
ax.set(xlabel="Pérdida de R² al permutar la familia de variables")
ax.legend(fontsize=8.5, loc="lower right")
guardar(fig, "F16_importancia_familias.png")

# ---------------------------------------------------------------- F17 error por fase
pf = pd.read_csv(TABLAS / "05_rmse_por_fase.csv")
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for nombre, c, ls in (("Persistencia", GRIS2, "--"), ("Mecanístico reducido", GRIS3, ":"), ("MLP propuesta", NARANJA, "-")):
    d = pf[(pf.horizonte_min == 30) & (pf.modelo == nombre)]
    ax.plot(d.posicion_ciclo, d.rmse, color=c, lw=2, ls=ls, marker="o", ms=4, label=nombre)
ax.set(xlabel="Posición en el ciclo en el instante de predicción (fracción)", ylabel="RMSE a 30 min (mg/L)", ylim=(0, None))
ax.legend(fontsize=8.5)
guardar(fig, "F17_error_por_fase_ciclo.png")

# ---------------------------------------------------------------- F18 R² mensual
rm = pd.read_csv(TABLAS / "05_r2_mensual.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for i, (h, c) in enumerate([(10, AZUL), (30, NARANJA)]):
    d = rm[rm.horizonte_min == h]
    x = np.arange(len(d)) + (i - .5) * .36
    ax.bar(x, d.r2, .34, color=c, label=f"{h} min")
    for xx, v in zip(x, d.r2):
        ax.text(xx, v + .015, f"{v:.2f}", ha="center", fontsize=8)
    ax.plot([-.5, len(d) - .5], [METAS[h]] * 2, color=c, ls="--", lw=1)
ax.set_xticks(range(len(d)), [f"{MESES[int(m[5:]) - 1]} {m[:4]}" for m in d.mes])
ax.set(ylabel="R² MLP propuesta", ylim=(0, 1))
ax.legend(fontsize=8.5, loc="upper left", ncol=2)
guardar(fig, "F18_r2_mensual_prueba.png")

# ---------------------------------------------------------------- F19 alerta temprana
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, h, l in zip(axes, (10, 30), "ab"):
    j = hs.index(h); yt = Y[:, j] > 2.0
    for nombre, c in (("LightGBM", AZUL), ("MLP propuesta", NARANJA)):
        pr, rc, umb = precision_recall_curve(yt, P[f"{h}|{nombre}"])
        ax.plot(rc, pr, color=c, lw=2, label=nombre)
        k = np.argmin(np.abs(umb - 2.0))
        ax.scatter(rc[k], pr[k], color=c, s=45, zorder=4, edgecolor=FONDO)
    pp = P[f"{h}|Persistencia"] > 2.0
    ax.scatter((pp & yt).sum() / yt.sum(), (pp & yt).sum() / pp.sum(), marker="s", color=GRIS, s=45, label="Persistencia", zorder=4)
    ax.axhline(yt.mean(), color=TINTA2, lw=.8, ls=":")
    ax.text(.98, yt.mean() + .02, "tasa base", ha="right", fontsize=8, color=TINTA2)
    ax.set(xlabel=f"Exhaustividad, {h} min", ylabel="Precisión", xlim=(0, 1), ylim=(0, 1.02))
    etiqueta(ax, f"({l})")
axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, .1))
fig.tight_layout()
guardar(fig, "F19_alerta_temprana.png")

# ---------------------------------------------------------------- F20 mecanístico vs red
a_, b_ = ini + 300, ini + 540
fig, ax = plt.subplots(figsize=(9, 3.4))
t = pd.to_datetime(fechas_te[a_:b_]) + pd.Timedelta(minutes=30)
ax.plot(t, Y[a_:b_, hs.index(30)], color=TINTA, lw=1.4, label="Observado")
ax.plot(t, P["30|Mecanístico reducido"][a_:b_], color=GRIS3, lw=1.5, ls="-.", label="Mecanístico reducido")
ax.plot(t, P["30|MLP propuesta"][a_:b_], color=NARANJA, lw=1.6, label="MLP propuesta")
ax.xaxis.set_major_formatter(fmt_fecha_hora())
ax.set(ylabel="PO₄ a 30 min (mg/L)")
ax.legend(fontsize=8.5, ncol=3, loc="upper left")
guardar(fig, "F20_mecanistico_vs_red.png")
