# Wine Quality | MP5 — Grupo #4

## Desarrollo completo: configuración, EDA, baseline, ratios, PCA, modelos avanzados y entrega final

Clasificación supervisada de vinos de calidad estándar (0) y superior (1). La métrica principal es F1 Macro. El proyecto cubre los momentos 1–6 de la guía: configuración, EDA, baseline, ingeniería de características, modelos avanzados y entrega final.

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


### Momento 5 — SVM, MLP y optimización (75–100 min)

Se compararon 533 configuraciones nuevas: 480 k-NN, 44 SVM y 9 MLP. Las 231 operaciones de ajuste reales de búsqueda incluyen preprocesamiento y búsqueda compartida de vecinos, ajustes completos de SVM/MLP y verificación de ocho candidatos k-NN. La auditoría agrupada añade tres ajustes. Se conservan las particiones de los momentos anteriores. El umbral se elige sobre las predicciones OOF de desarrollo, por lo cual la métrica optimizada puede ser optimista; el holdout no se consulta en este avance. No se usan etiquetas externas.

Se exploran SVM lineal, RBF y polinomial de grado 2 con C, gamma y pesos de clase; MLP con ReLU/tanh, distintas capas, alpha, learning rate, lotes de 64, parada temprana y pesos de muestra. La parada temprana de scikit-learn usa accuracy en la validación interna del MLP, pero la selección externa usa F1 Macro. En k-NN se exploran StandardScaler/RobustScaler, PCA 95% o sin PCA, ratios, Manhattan/Euclidiana, k y ponderación inversa a distancia elevada a 1, 2 o 3. Los candidatos con caché son exploratorios: solo se elige un pipeline verificado directamente y sin alerta de convergencia.

| Familia | F1 Macro OOF | F1 umbral nativo | Accuracy | Precision 1 | Recall 1 | F1 clase 1 | Ajuste medio (ms) | Inferencia media (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-NN | 0.8504 | 0.8323 | 0.9192 | 0.7411 | 0.7568 | 0.7489 | 11.89 | 70.71 |
| SVM | 0.8206 | 0.7991 | 0.9062 | 0.7179 | 0.6767 | 0.6967 | 57.28 | 28.21 |
| MLP | 0.8155 | 0.7957 | 0.8997 | 0.6783 | 0.7039 | 0.6909 | 295.56 | 1.65 |

Candidato provisional: **k-NN k=100 euclidean 1/d^3 ratios=1 robust=0 PCA=0.95**, umbral 0.4000, F1 Macro OOF **0.8504**. Mejora de 6.15 puntos porcentuales sobre el baseline. Auditoría por grupos con umbral fijo: 0.8105. La auditoría no es el leaderboard. No se genera aún modelo definitivo ni submission.

#### Tiempos reales de la ejecución acumulada

| Actividad | Segundos |
|---|---:|
| EDA | 2.1834 |
| Momento 3 | 1.5751 |
| Momento 4 | 9.4232 |
| busqueda knn (momento 5) | 8.9254 |
| busqueda svm (momento 5) | 17.1440 |
| busqueda mlp (momento 5) | 9.3461 |
| verificacion knn (momento 5) | 2.1380 |
| auditoria grupos (momento 5) | 0.2040 |
| total (momento 5) | 38.0514 |
| Notebook completo, incluido kernel | 54.8678 |

La búsqueda de k-NN reutiliza distancias. Los registros `cache_vecinos` miden el voto individual, mientras los tiempos compartidos están en `resultados/avanzado_cache_vecinos.csv`; no se deben contar repetidamente. El ajuste e inferencia de candidatos elegibles sí miden el pipeline completo. Las duraciones de los modelos no equivalen a tiempo humano.

#### Archivos nuevos

- `modelado.py`: construcción de modelos, búsqueda, optimización de umbral y auditoría.
- `resultados/avanzado_metricas.csv`: métricas por configuración.
- `resultados/avanzado_folds.csv`: métricas y tiempos por fold.
- `resultados/avanzado_cache_vecinos.csv`: costo compartido de las búsquedas de vecinos.
- `resultados/avanzado_configuraciones.json`: parámetros reproducibles de todos los candidatos.
- `resultados/resumen_avanzado.json`: candidato provisional y auditoría.
- `resultados/avanzado_comparacion.png`: progresión y matriz de confusión.
- `resultados/tiempo_notebook_avance4.json`: tiempo total de notebook.



### Momento 6 — Selección definitiva y entregables (100–120 min)

El ganador se fija por el mayor F1 Macro OOF entre los candidatos elegibles verificados en el momento 5. No se amplía la búsqueda ni se ajusta nuevamente el umbral. Se audita con las 1.040 muestras apartadas de la búsqueda, se reajusta con las 5.197 muestras etiquetadas y se predice el test de 1.300 filas. No se utilizan etiquetas externas ni se evalúa el conjunto oculto.

**Modelo definitivo:** k-NN con k=100, distancia euclidiana, pesos inversos cúbicos, tres ratios, StandardScaler, PCA con retención mínima del 95% y umbral 0,40. El ajuste final conserva 9 componentes y 95.84% de varianza. Si la distancia es cero, únicamente votan los vecinos coincidentes. La predicción superior ocurre cuando el voto ponderado normalizado alcanza el umbral. El voto no es una probabilidad calibrada.

| Métrica | OOF (selección) | Auditoría final |
|---|---:|---:|
| F1 Macro | 0.8504 | 0.8659 |
| Accuracy | 0.9192 | 0.9279 |
| Precision clase superior | 0.7411 | 0.7725 |
| Recall clase superior | 0.7568 | 0.7771 |
| F1 clase superior | 0.7489 | 0.7748 |

La auditoría agrupada del momento 5 obtiene F1 Macro 0.8105. Holdout separado del ajuste y selección de esta secuencia; ya fue consultado en el prototipo inicial y el EDA global usó train completo. No equivale a una evaluación ciega ni al leaderboard. El prototipo anterior llegó a 0.8701 en la misma partición con otro modelo; no se utiliza esa observación para cambiar la selección por OOF. La diferencia no prueba que uno u otro generalice mejor al test oculto.

#### Submission

`entregas/submission_Grupo#4.csv`: 1300 observaciones, 1301 líneas incluyendo encabezado, dos columnas (`id`, `prediction`), 1073 predicciones estándar y 227 superiores. Se verificaron identificadores, orden, ausencia de nulos, clases binarias, hash y coincidencia con las predicciones del modelo recargado. El resultado del leaderboard no se conoce.

#### Tiempos del cierre

| Operación | Segundos |
|---|---:|
| carga_y_seleccion_s | 0.015912 |
| ajuste_auditoria_s | 0.012218 |
| inferencia_auditoria_s | 0.060878 |
| metricas_auditoria_s | 0.006115 |
| ajuste_final_s | 0.015671 |
| inferencia_test_s | 0.083178 |
| guardar_modelo_csv_s | 0.025007 |
| verificacion_csv_modelo_s | 0.090516 |
| reportes_s | 0.197164 |
| total_momento6_s | 0.526496 |

#### Tiempos acumulados (sin arranque de kernel)

| Momento | Segundos |
|---|---:|
| momento2_eda | 2.183418 |
| momento3_baseline | 1.575110 |
| momento4_feature_pca | 9.423234 |
| momento5_avanzado | 38.051365 |
| momento6_cierre | 0.526496 |
| **Total computacional** | **51.759622** |

La evidencia del avance 4 se conserva sin repetir la búsqueda: 54.868 s de notebook anterior y 4.062 s de ejecución de las celdas de cierre en otro kernel (suma 58.930 s). No es una ejecución íntegra nueva ni tiempo de trabajo humano. Los tiempos de configuración humana y del pitch no se midieron. Total de 593 configuraciones evaluadas por etapa, contando repeticiones entre etapas (6 + 54 + 533).

#### Ejecutar el resultado final

Desde la raíz del repositorio, después de instalar las dependencias:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

El modelo ya está guardado: no se necesita volver a entrenar. Para comprobar el CSV:

```powershell
.\.venv\Scripts\python.exe verificar.py
```

Para reproducir únicamente el cierre sin repetir la búsqueda:

```powershell
.\.venv\Scripts\python.exe finalizar.py
```

El notebook conserva los resultados ejecutados de los momentos 1–5 y las celdas ejecutadas del cierre. Si se desea reproducir toda la búsqueda, reiniciar el kernel y ejecutar todas las celdas. El código de la aplicación se compiló y se verificaron sus datos y el predictor serializado; Streamlit no estaba instalado en el entorno de ejecución, por lo que falta probar visualmente la app localmente antes de subirla. GitHub y la asignación de roles corresponden al equipo.

#### Archivos del cierre

- `app/app.py`: simulador, EDA, PCA, métricas y tiempos.
- `modelos/modelo.joblib`, `modelos/metricas.csv`, `modelos/resumen.json`: modelo y datos del tablero.
- `entregas/submission_Grupo#4.csv`: entrega oficial del grupo.
- `finalizar.py`, `verificar.py`: reproducibilidad y verificación.
- `resultados/resumen_final.json`, `resultados/tiempos_finales.csv`, `resultados/matriz_confusion_final.png`, `resultados/tiempo_notebook_avance5.json`: evidencias del cierre.
