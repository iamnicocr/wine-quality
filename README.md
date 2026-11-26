# Wine Quality | MP5 — Grupo #4

## Avance 3: configuración, EDA, baseline, ratios y PCA

Clasificación supervisada de vinos de calidad estándar (0) y superior (1). La métrica principal es F1 Macro. Este avance acumulado corresponde a los minutos 0–75 de la guía: configuración inicial, EDA, baseline k-NN e ingeniería de características. No se generan predicciones del test público.

### Datos

| Conjunto | Filas | Columnas |
|---|---:|---:|
| Entrenamiento público | 5197 | 13 |
| Prueba pública | 1300 | 12 |

Las 11 propiedades fisicoquímicas son numéricas. Se excluye `id` de los predictores y `target` constituye la salida. En entrenamiento hay 4369 muestras estándar y 828 superiores. Los faltantes son 0 en entrenamiento y 0 en prueba (solo predictores). Se identifican 807 filas adicionales con perfiles repetidos y 1 perfiles con etiquetas diferentes. Se conservan los registros y los extremos; no hay evidencia suficiente para eliminarlos.

### Decisiones iniciales

- Las escalas son heterogéneas: se ajustará StandardScaler dentro de cada fold para prevenir fuga de datos.
- La distribución de clases justifica F1 Macro en lugar de seleccionar por accuracy.
- Se reserva la validación independiente y se usará StratifiedKFold para comparar modelos. Los duplicados justifican una auditoría por grupos adicional.
- Se compararán los métodos autorizados por la guía; el baseline k-NN ya se evalúa en este avance.

### Ejecutar

Python 3.11–3.13. Desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook notebooks/01_eda_y_modelado.ipynb
```

En Linux/macOS usar `.venv/bin/python`. El notebook ya contiene los resultados ejecutados. Puede volver a ejecutarse con Restart & Run All.

### Evidencia

- `notebooks/01_eda_y_modelado.ipynb`: notebook acumulado ejecutado, con tablas y gráficos.
- `resultados/`: estadísticas, distribución de clases, correlaciones, gráficos y tiempos reales de EDA.
- `docs/TRAZABILIDAD.md`: registro acumulado y planificación por momentos oficiales; no simula commits previos.
- `docs/ROLES.md`: asignación propuesta por completar con nombres reales.

### Resultados del baseline (validación cruzada estratificada)

Semilla 42, 3 folds sobre 4.157 observaciones; 1.040 muestras de entrenamiento quedan reservadas para la auditoría final. Pipeline de imputación por mediana, StandardScaler y k-NN con voto uniforme, ajustado dentro de cada fold. Los 11 predictores originales se mantienen.

| k | Distancia | F1 Macro OOF | Accuracy | Precision (1) | Recall (1) | Ajuste medio (s) | Inferencia media (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | manhattan | 0.7889 | 0.8879 | 0.6512 | 0.6375 | 0.0060 | 0.0423 |
| 7 | manhattan | 0.7883 | 0.8939 | 0.6977 | 0.5891 | 0.0070 | 0.0485 |
| 3 | euclidean | 0.7865 | 0.8869 | 0.6491 | 0.6314 | 0.0112 | 0.0327 |
| 7 | euclidean | 0.7844 | 0.8915 | 0.6867 | 0.5861 | 0.0061 | 0.0309 |
| 5 | manhattan | 0.7822 | 0.8889 | 0.6706 | 0.5937 | 0.0062 | 0.0483 |
| 5 | euclidean | 0.7818 | 0.8901 | 0.6808 | 0.5831 | 0.0063 | 0.0288 |

El baseline seleccionado es k=3 con Manhattan: F1 Macro OOF **0.7889**. La referencia de predecir siempre la clase 0 tiene F1 Macro de 0.4567, pese a su accuracy de 0.8408. La auditoría adicional separando perfiles idénticos obtiene F1 Macro 0.7819. No corresponde a la prueba oculta.

### Tiempos medidos en el avance 2 (registro histórico)

| Actividad | Segundos |
|---|---:|
| EDA (reejecución acumulada) | 2.3257 |
| Búsqueda baseline (6 configuraciones, 18 ajustes) | 0.9428 |
| Auditoría adicional por grupos (3 ajustes) | 0.4495 |
| Momento 3 completo | 1.6591 |
| Notebook acumulado con kernel | 8.1535 |

Los tiempos corresponden al entorno de ejecución; no son minutos de trabajo humano. El detalle por fold está en `resultados/baseline_folds.csv`. La comparación de modelos está en `resultados/baseline_metricas.csv`, la matriz de confusión y la gráfica en `resultados/baseline_comparacion.png`, y el resumen en `resultados/resumen_baseline.json`.

### Momento 4 — Feature engineering y PCA (50–75 min)

Se crean tres variables: `ratio_acidez = fixed_acidity / volatile_acidity`, `ratio_so2 = free_sulfur_dioxide / total_sulfur_dioxide` y `ratio_alcohol_densidad = alcohol / density`. Estas relaciones tienen interpretación fisicoquímica; no se supone causalidad. Se conservan los atributos originales. Las divisiones por cero generan faltantes que se imputan dentro del fold. Se comparan 11 y 14 atributos, con/sin PCA y con retención mínima de 91% o 95%. El pipeline ajusta imputación, escalado y PCA exclusivamente sobre cada fold de entrenamiento.

Se mantienen los mismos 4.157 registros y tres folds. Se incluye una comparación controlada con k=3, Manhattan y voto uniforme, y una exploración de k=3, 7, 15, 50 con voto ponderado y las dos distancias. No se ajusta el umbral aún. Se evaluaron 54 configuraciones (incluida la repetición del baseline para control), 162 ajustes.

| Ratios | PCA mínimo | Mejor F1 Macro OOF | Componentes medios | Varianza media | Ajuste (ms) | Inferencia (ms) |
|---|---:|---:|---:|---:|---:|---:|
| No | 95% | 0.8262 | 9 | 97.58% | 6.48 | 53.59 |
| No | Sin PCA | 0.8259 | 11 | 100.00% | 6.00 | 64.80 |
| Sí | 91% | 0.8259 | 8 | 92.25% | 9.20 | 36.16 |
| Sí | 95% | 0.8229 | 9 | 95.84% | 7.99 | 36.22 |
| Sí | Sin PCA | 0.8207 | 14 | 100.00% | 8.69 | 77.69 |
| No | 91% | 0.8162 | 8 | 94.54% | 6.85 | 35.75 |

En el control de k=3, Manhattan y voto uniforme, los ratios elevan el F1 de 0.7889 a 0.7934 sin PCA; ratios + PCA 95% alcanza 0.7950. No obstante, el mejor candidato general de esta etapa utiliza las 11 variables originales, PCA 95% (9 componentes y 97.58% de varianza media), k=50, Manhattan y ponderación por distancia. Su F1 Macro es 0.8262, +3.73 puntos porcentuales respecto al baseline. El candidato sin PCA y con la misma configuración obtiene 0.8259; la diferencia es muy pequeña y no prueba superioridad estadística. PCA reduce las dimensiones y en esta ejecución disminuye la inferencia media. Se conservan ambas alternativas para el siguiente momento.

El candidato tiene accuracy 0.9148, precision de clase superior 0.7928, recall 0.6299 y F1 de clase superior 0.7020. La auditoría complementaria agrupando perfiles iguales obtiene 0.7960 de F1 Macro. El recall es ligeramente menor al baseline, por lo que se evaluarán umbrales en el momento siguiente. No es rendimiento del test oculto.

### Tiempos de la ejecución acumulada actual

| Actividad | Segundos |
|---|---:|
| EDA | 2.0601 |
| Momento 3 completo | 1.6135 |
| Búsqueda feature engineering y PCA | 8.2314 |
| Auditoría agrupada del candidato | 0.4724 |
| Momento 4 completo (incluye búsqueda y auditoría) | 9.1195 |
| Notebook acumulado con kernel | 16.5055 |

Resultados: `feature_pca_metricas.csv`, `feature_pca_folds.csv`, `feature_pca_control.csv`, `ratios_estadisticas.csv`, `pca_varianza.png`, `pca_comparacion.png`, `resumen_feature_pca.json`. Los tiempos son mediciones reales del entorno, no duración humana ni marcas de commits.

### Siguiente momento

SVM y MLP (75–100 min): comparar modelos avanzados y optimizar parámetros y umbral sobre desarrollo. No se incluye todavía app, modelo final ni submission. El holdout de 1.040 observaciones sigue reservado.
