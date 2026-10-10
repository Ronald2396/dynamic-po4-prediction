# Datos

El conjunto de datos no se incluye en el repositorio. Para reproducir la investigación:

1. Descarga el dataset público *Wastewater Treatment Plant Data for Nutrient Removal System* (Mohammadi et al., 2024) desde Mendeley Data: <https://data.mendeley.com/datasets/34rpmsxc4z> (DOI 10.17632/34rpmsxc4z).
2. Guarda el archivo CSV como `data/dataset.csv`.

## Formato esperado

525 600 filas, una cada 2 minutos, entre el 2021-08-01 y el 2023-07-31, con las siguientes columnas:

| Columna | Descripción | Unidad |
|---|---|---|
| `date` | Fecha y hora (UTC) | — |
| `T1_PO4` | **Fosfato en el tanque biológico 1 (variable objetivo)** | mg/L |
| `T1_NH4` | Amonio en el tanque 1 | mg/L |
| `T1_O2` | Oxígeno disuelto en el tanque 1 | mg/L |
| `IN_Q` | Caudal afluente a la biología | m³/h |
| `IN_METAL_Q` | Dosificación de hierro a la biología | L/h |
| `METAL_Q` | Dosificación de PAC al decantador secundario | L/h |
| `TEMPERATURE` | Temperatura del tanque | °C |
| `MAX_CF` | Señal operacional de capacidad | % |
| `PROCESSPHASE_INLET` | Fase de entrada de la operación alternada (1/2) | — |
| `PROCESSPHASE_OUTLET` | Fase de salida de la operación alternada (1/2) | — |
