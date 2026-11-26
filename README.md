# Wine Quality | MP5 — Grupo #4

## Avance 1: configuración y exploración de datos

Clasificación supervisada de vinos de calidad estándar (0) y superior (1). La métrica principal es F1 Macro. Este avance corresponde a los minutos 0–30 de la guía: configuración inicial y EDA. Aún no se entrenan modelos ni se generan predicciones.

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
- Se compararán los métodos autorizados por la guía; la fase siguiente será el baseline k-NN.

### Ejecutar

Python 3.11–3.13. Desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook notebooks/01_eda_y_modelado.ipynb
```

En Linux/macOS usar `.venv/bin/python`. El notebook ya contiene los resultados ejecutados. Puede volver a ejecutarse con Restart & Run All.

### Evidencia

- `notebooks/01_eda_y_modelado.ipynb`: notebook inicial ejecutado, con tablas y gráficos.
- `resultados/`: estadísticas, distribución de clases, correlaciones, gráficos y tiempos reales de EDA.
- `docs/TRAZABILIDAD.md`: alcance actual y planificación por momentos oficiales; no simula commits previos.
- `docs/ROLES.md`: asignación propuesta por completar con nombres reales.

### Tiempos medidos

| Actividad | Segundos |
|---|---:|
| Carga y validación | 0.0165 |
| Control de calidad | 0.0125 |
| Distribución de clases | 0.1433 |
| Distribuciones y extremos | 1.2565 |
| Correlaciones y gráficas | 0.7650 |
| Guardado de tablas | 0.0033 |
| Total medido dentro del notebook | 2.2475 |
| Ejecución completa con kernel | 5.0959 |

Tiempos del entorno de prueba, no equivalentes al tiempo humano del equipo. En los siguientes avances se registrarán ajuste, inferencia y búsqueda por modelo.
