# Predicción dinámica de fosfato (PO₄) con redes neuronales en una PTAR

Repositorio de la tesis **"Desarrollo de un modelo predictivo basado en redes neuronales para la predicción dinámica a corto plazo de la concentración de fosfato en una planta de tratamiento de aguas residuales"** (Escuela Profesional de Ingeniería Química, Universidad Nacional de Ingeniería, Lima, Perú).

La red neuronal predice la concentración de fosfato del tanque biológico **10 y 30 minutos antes** de que ocurra. Usa solo las señales que la planta ya registra en su sistema SCADA.

| Horizonte | Meta de la tesis | Red neuronal propuesta (R² en prueba) | Validación de 1 año, 4 trimestres (IC 95 %) |
|---|---|---|---|
| 10 min | R² > 0.70 | **0.822** | **0.876** [0.860–0.888] |
| 30 min | R² > 0.60 | **0.720** | **0.771** [0.747–0.790] |

**Documento completo de la tesis** (métodos, resultados y discusión): [tesis/README.md](tesis/README.md)

---

## ¿Cómo funciona?

La planta de Agtrup (Dinamarca) opera en **ciclos alternados**: el agua residual entra al tanque biológico por fases y cada ciclo dura entre 1 y 2.5 horas. El fosfato sube cuando entra el afluente y baja durante la aireación, cuando las bacterias acumuladoras de fósforo (PAO) lo captan. Casi la mitad de la variabilidad del PO₄ ocurre dentro de cada ciclo.

![Ciclo operacional](figuras/F03_ventana_ciclo_operacional.png)

El aporte central de esta investigación es darle a la red un **"reloj de fase"**: variables que le indican en qué punto del ciclo está el reactor y cuánto duraron las fases anteriores. Con esa información y la historia reciente del proceso, una red neuronal multicapa (MLP) anticipa la subida y la bajada de cada ciclo. La persistencia, en cambio, queda desfasada.

![Comparación de modelos](figuras/F08_comparacion_r2.png)

## Metodología

![Diagrama metodológico](figuras/F01_diagrama_metodologia.png)

1. **Datos.** Se usan 525 600 registros SCADA cada 2 min (2021–2023). Son públicos ([Mohammadi et al., 2024](https://data.mendeley.com/datasets/34rpmsxc4z)).
2. **Variables.** Se calculan 153 variables causales agrupadas en 8 familias (reloj de fase, historia de PO₄, NH₄, O₂, caudal, dosificación, temperatura y calendario). Una prueba formal verifica que ninguna use información futura.
3. **Validación temporal.** La partición es cronológica 70/15/15, con una purga de 60 min en cada frontera, más una validación por origen móvil en 4 trimestres.
4. **Red neuronal propuesta.** Es un ensamble de 5 MLP (3 × 512 neuronas, GELU, LayerNorm), con salida multihorizonte (10, 30 y 60 min), entrenada con AdamW, pérdida de Huber y detención temprana.
5. **Modelos de comparación.**
   - Persistencia.
   - Naive de ciclo previo.
   - Modelo mecanístico reducido inspirado en ASM2d.
   - LightGBM.
   - Redes secuenciales GRU y TCN.
6. **Evaluación.**
   - Métricas: R², r de Pearson, RMSE, MAE, sesgo e índice de habilidad frente a persistencia.
   - IC 95 % por bootstrap de bloques diarios.
   - Prueba de Diebold–Mariano.
   - Alerta temprana de PO₄ > 2 mg/L.

## Estructura del repositorio

```
├── data/                     # dataset.csv (no versionado; ver data/README.md)
├── src/po4_prediccion/       # módulo con el código reutilizable
│   ├── datos.py              #   carga, 153 variables causales, particiones
│   ├── red_neuronal.py       #   MLP propuesta y entrenamiento
│   ├── referencias.py        #   persistencia, naive, mecanístico, LightGBM
│   └── metricas.py           #   R², RMSE, MAE, SS, bootstrap, Diebold–Mariano
├── scripts/                  # pipeline de la tesis, en orden de ejecución
│   ├── 01_caracterizacion_datos.py
│   ├── 02_verificacion_fuga.py
│   ├── 03_modelos_referencia.py
│   ├── 04_red_neuronal.py
│   ├── 05_evaluacion.py
│   ├── 06_backtest_estacional.py
│   ├── 07_ablacion_entradas.py
│   ├── 08_explicabilidad.py
│   └── 09_figuras.py
├── notebooks/
│   └── redes_secuenciales_gru_tcn_colab.ipynb   # GRU/TCN en GPU (Google Colab)
├── resultados/
│   ├── tablas/               # todas las tablas de resultados (CSV)
│   └── colab/                # predicciones de GRU/TCN obtenidas en Colab
├── figuras/                  # 21 figuras de la tesis (PNG, 300 dpi)
└── tesis/README.md           # métodos, resultados y discusión
```

## Reproducir los resultados

Requisitos: Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Descarga el dataset en `data/dataset.csv` (ver [data/README.md](data/README.md)) y ejecuta los scripts en orden:

```bash
uv run python scripts/01_caracterizacion_datos.py
```

```bash
uv run python scripts/02_verificacion_fuga.py
```

```bash
uv run python scripts/03_modelos_referencia.py
```

```bash
uv run python scripts/04_red_neuronal.py
```

```bash
uv run python scripts/05_evaluacion.py
```

```bash
uv run python scripts/06_backtest_estacional.py
```

```bash
uv run python scripts/07_ablacion_entradas.py
```

```bash
uv run python scripts/08_explicabilidad.py
```

```bash
uv run python scripts/09_figuras.py
```

- Todo el pipeline tarda ≈1.5 h en una CPU de 16 núcleos y no necesita GPU.
- Las predicciones de GRU y TCN ya están en `resultados/colab/`. Para reentrenarlas, abre el notebook en Google Colab con GPU.
- Las semillas están fijadas (42, 7, 123, 2024 y 31415).

## Datos

Mohammadi, E. et al. (2024). *Wastewater Treatment Plant Data for Nutrient Removal System*. Mendeley Data. DOI [10.17632/34rpmsxc4z](https://data.mendeley.com/datasets/34rpmsxc4z). PTAR de Agtrup (BlueKolding, Kolding, Dinamarca).
