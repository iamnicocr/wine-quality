from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelado import COLUMNAS, scores

st.set_page_config(page_title='Wine Quality | Grupo 4', page_icon='🍇', layout='wide')

@st.cache_resource
def cargar_modelo():
    return joblib.load(ROOT/'modelos/modelo.joblib')

@st.cache_data
def cargar_datos():
    a = pd.read_csv(ROOT/'datos_estudiantes/train_public.csv')
    t = pd.read_csv(ROOT/'modelos/metricas.csv')
    r = json.loads((ROOT/'modelos/resumen.json').read_text(encoding='utf-8'))
    return a,t,r

try:
    a,t,r = cargar_datos()
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error('Ejecuta python finalizar.py desde la raíz del proyecto.'); st.stop()

st.title('Wine Quality | Grupo 4')
st.caption('Clasificación de calidad del vino a partir de 11 propiedades fisicoquímicas.')
c1,c2,c3 = st.columns(3)
c1.metric('F1 Macro · auditoría final', f"{r['holdout']['f1_macro']:.2%}")
c2.metric('Precisión · calidad superior', f"{r['holdout']['precision_1']:.2%}")
c3.metric('Recall · calidad superior', f"{r['holdout']['recall_1']:.2%}")
st.write('Modelo seleccionado:', r['ganador'])

prediccion, exploracion, comparacion = st.tabs(['Predicción','Exploración de datos','Comparación de modelos'])
with prediccion:
    st.subheader('Simulador de calidad')
    etiquetas = ['Acidez fija (g/dm³)','Acidez volátil (g/dm³)','Ácido cítrico (g/dm³)','Azúcar residual (g/dm³)','Cloruros (g/dm³)','SO₂ libre (mg/dm³)','SO₂ total (mg/dm³)','Densidad (g/cm³)','pH','Sulfatos (g/dm³)','Alcohol (% vol.)']
    with st.form('vino'):
        cols = st.columns(3); entrada = {}
        for i,c in enumerate(COLUMNAS):
            with cols[i%3]:
                entrada[c] = st.number_input(etiquetas[i], min_value=0.0, value=float(a[c].median()), format='%.5f' if c=='density' else '%.3f', step=.0001 if c=='density' else .01)
        ejecutar = st.form_submit_button('Predecir calidad',type='primary')
    if ejecutar:
        x = pd.DataFrame([entrada],columns=COLUMNAS)
        s = float(scores(modelo['modelo'],x)[0]); pred = int(s>=modelo['umbral'])
        fuera = [c for c in COLUMNAS if entrada[c]<a[c].min() or entrada[c]>a[c].max()]
        if fuera: st.warning('Hay valores fuera del rango de entrenamiento: '+', '.join(fuera))
        if pred: st.success('Calidad superior')
        else: st.info('Calidad estándar')
        st.metric('Voto ponderado a favor de calidad superior', f'{s:.2%}')
        st.caption(f"Umbral: {modelo['umbral']:.2f}. El voto normalizado no es una probabilidad calibrada ni una garantía de certeza.")
with exploracion:
    c1,c2 = st.columns(2)
    with c1:
        st.subheader('Distribución de clases')
        st.image(str(ROOT/'resultados/distribucion_clases.png'))
        st.subheader('Variables según calidad')
        st.image(str(ROOT/'resultados/variables_por_clase.png'))
    with c2:
        st.subheader('Correlaciones')
        st.image(str(ROOT/'resultados/correlaciones.png'))
        st.subheader('Varianza explicada por PCA')
        st.image(str(ROOT/'resultados/pca_varianza.png'))
        st.caption(f"Modelo final: {r['componentes']} componentes, {r['varianza_pca']:.2%} de varianza retenida.")
with comparacion:
    st.subheader('Resultados de validación cruzada')
    st.caption('Mismas particiones estratificadas. Umbral optimizado en OOF para candidatos avanzados; el test oculto no interviene.')
    st.bar_chart(t.set_index('etapa').f1_macro)
    st.dataframe(t,hide_index=True,use_container_width=True)
    st.subheader('Matriz de confusión · auditoría final')
    st.image(str(ROOT/'resultados/matriz_confusion_final.png'))
    st.caption(f"Auditoría por grupos de duplicados: F1 Macro {r['auditoria_grupos']['f1_macro']:.2%}. {r['nota_validacion']}")
    st.subheader('Tiempos computacionales')
    st.dataframe(pd.DataFrame(list(r['tiempos_acumulados_s'].items()), columns=['Momento','Segundos']),hide_index=True,use_container_width=True)
    st.metric('Total acumulado (sin arranque de kernel)', f"{r['tiempo_computacional_acumulado_s']:.3f} s")
    st.caption('Tiempos del entorno de ejecución, no duración del trabajo del equipo.')
    st.download_button('Descargar predicciones', (ROOT/'entregas/submission_Grupo#4.csv').read_bytes(), file_name='submission_Grupo#4.csv', mime='text/csv')
