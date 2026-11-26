"""Cierre del MP5: usa únicamente el ganador verificado del momento 5."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from modelado import COLUMNAS, SEMILLA, cargar, crear, ajustar, scores, metricas, nombre

ROOT = Path(__file__).resolve().parent
ARCHIVO = 'submission_Grupo#4.csv'

def leer_json(nombre_archivo):
    return json.loads((ROOT / 'resultados' / nombre_archivo).read_text(encoding='utf-8'))

def finalizar():
    inicio = perf_counter()
    tiempos = {}
    t = perf_counter()
    previo = leer_json('resumen_avanzado.json')
    tabla = pd.read_csv(ROOT / 'resultados/avanzado_metricas.csv')
    ganador = tabla.loc[tabla.elegible].sort_values('f1_macro', ascending=False).iloc[0]
    c = previo['configuracion']
    umbral = float(ganador.umbral)
    assert nombre(c) == ganador.modelo == previo['ganador']['modelo']
    train, test = cargar()
    X = train[COLUMNAS]
    y = train.target.to_numpy()
    dev, hold = train_test_split(np.arange(len(X)), test_size=.2, random_state=SEMILLA, stratify=y)
    assert len(dev) == 4157 and len(hold) == 1040
    tiempos['carga_y_seleccion_s'] = perf_counter()-t
    # No se cambia ni el candidato ni el umbral después de la auditoría.
    with threadpool_limits(limits=1):
        t = perf_counter()
        modelo_auditoria = crear(c)
        ajustar(modelo_auditoria, X.iloc[dev], y[dev], c['familia']=='MLP' and c['balance'])
        tiempos['ajuste_auditoria_s'] = perf_counter()-t
        t = perf_counter()
        s = scores(modelo_auditoria, X.iloc[hold])
        p = (s >= umbral).astype(int)
        tiempos['inferencia_auditoria_s'] = perf_counter()-t
        t = perf_counter()
        audit = metricas(y[hold], p, s)
        audit['matriz_confusion'] = confusion_matrix(y[hold], p, labels=[0,1]).tolist()
        tiempos['metricas_auditoria_s'] = perf_counter()-t
        # El reajuste final incluye toda la información etiquetada permitida.
        t = perf_counter()
        modelo = crear(c)
        ajustar(modelo, X, y, c['familia']=='MLP' and c['balance'])
        tiempos['ajuste_final_s'] = perf_counter()-t
        t = perf_counter()
        s_test = scores(modelo, test[COLUMNAS])
        pred = (s_test >= umbral).astype(int)
        tiempos['inferencia_test_s'] = perf_counter()-t
        t = perf_counter()
        carpeta = ROOT / 'modelos'; carpeta.mkdir(exist_ok=True)
        (ROOT / 'entregas').mkdir(exist_ok=True)
        paquete = dict(modelo=modelo, umbral=umbral, configuracion=c, tipo_score='Voto ponderado normalizado (no calibrado)')
        joblib.dump(paquete, carpeta/'modelo.joblib', compress=3)
        salida = ROOT/'entregas'/ARCHIVO
        pd.DataFrame({'id':test.id,'prediction':pred}).to_csv(salida,index=False)
        tiempos['guardar_modelo_csv_s'] = perf_counter()-t
        t = perf_counter()
        ver = pd.read_csv(salida)
        recargado = joblib.load(carpeta/'modelo.joblib')
        assert list(ver.columns)==['id','prediction'] and len(ver)==1300
        assert ver.id.equals(test.id) and ver.id.is_unique and not ver.isna().any().any()
        assert set(ver.prediction.unique()).issubset({0,1}) and np.isfinite(s_test).all()
        assert np.array_equal(ver.prediction.to_numpy(), (scores(recargado['modelo'],test[COLUMNAS])>=recargado['umbral']).astype(int))
        assert len(salida.read_text().splitlines())==1301
        tiempos['verificacion_csv_modelo_s'] = perf_counter()-t
    # Resumen para dashboard: la comparación OOF conserva los tiempos del avance 4.
    t = perf_counter()
    base = leer_json('resumen_baseline.json'); fe = leer_json('resumen_feature_pca.json')
    eda = leer_json('resumen_eda.json')
    filas = []
    for etiqueta, m in [('Baseline',base['mejor']),('Feature Engineering / PCA',fe['mejor'])]:
        filas.append(dict(etapa=etiqueta,modelo=f"k-NN k={m['k']} {m['distancia']}",umbral=.5,**{k:m[k] for k in ['f1_macro','accuracy','precision_1','recall_1','f1_1','ajuste_medio_s','inferencia_media_s']}))
    for _,m in tabla.loc[tabla.elegible].sort_values('f1_macro',ascending=False).groupby('familia',sort=False).head(1).iterrows():
        filas.append(dict(etapa=m.familia,modelo=m.modelo,**{k:m[k] for k in ['umbral','f1_macro','accuracy','precision_1','recall_1','f1_1','ajuste_medio_s','inferencia_media_s']}))
    pd.DataFrame(filas).to_csv(carpeta/'metricas.csv',index=False)
    pca = modelo.named_steps['pca']
    resumen = dict(ganador=ganador.modelo,configuracion=c,umbral=umbral,cv_mejor=float(ganador.f1_macro),cv_baseline=base['mejor']['f1_macro'],cv_feature_pca=fe['mejor']['f1_macro'],holdout=audit,auditoria_grupos=previo['auditoria_grupos'],n_dev=len(dev),n_holdout=len(hold),n_train_final=len(train),n_test=len(test),predicciones_0=int((pred==0).sum()),predicciones_1=int((pred==1).sum()),componentes=int(pca.n_components_) if pca!='passthrough' else 14 if c['ratios'] else 11,varianza_pca=float(pca.explained_variance_ratio_.sum()) if pca!='passthrough' else 1.,sha256=hashlib.sha256(salida.read_bytes()).hexdigest(),configuraciones_por_etapa=[6,54,533],nota_validacion='Holdout separado del ajuste y selección de esta secuencia; ya fue consultado en el prototipo inicial y el EDA global usó train completo. No equivale a una evaluación ciega ni al leaderboard.')
    # Imágenes de cierre, independientes de Streamlit.
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(5,4)); mat=np.array(audit['matriz_confusion']); ax.imshow(mat,cmap='Blues'); ax.set(xlabel='Predicción',ylabel='Clase real',xticks=[0,1],yticks=[0,1],title='Auditoría final (1.040 muestras)')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>mat.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(ROOT/'resultados/matriz_confusion_final.png',dpi=150); plt.close(fig)
    tiempos['reportes_s']=perf_counter()-t
    tiempos['total_momento6_s']=perf_counter()-inicio
    resumen['tiempos_s']=tiempos
    resumen['tiempos_acumulados_s']={'momento2_eda':eda['tiempos_s']['total_eda_s'],'momento3_baseline':base['tiempo_momento3_s'],'momento4_feature_pca':fe['tiempo_momento4_s'],'momento5_avanzado':previo['tiempos_s']['total'],'momento6_cierre':tiempos['total_momento6_s']}
    resumen['tiempo_computacional_acumulado_s']=sum(resumen['tiempos_acumulados_s'].values())
    (carpeta/'resumen.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'resultados/resumen_final.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame([{'operacion':k,'segundos':v} for k,v in tiempos.items()]).to_csv(ROOT/'resultados/tiempos_finales.csv',index=False)
    return resumen

if __name__=='__main__':
    print(json.dumps(finalizar(),indent=2,ensure_ascii=False))
