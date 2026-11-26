"""Comprueba el submission contra el modelo serializado sin reentrenarlo."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelado import ROOT, COLUMNAS, cargar, scores

def verificar():
    _, test = cargar()
    ruta = ROOT/'entregas/submission_Grupo#4.csv'
    csv = pd.read_csv(ruta)
    paquete = joblib.load(ROOT/'modelos/modelo.joblib')
    r = json.loads((ROOT/'modelos/resumen.json').read_text(encoding='utf-8'))
    assert csv.columns.tolist()==['id','prediction'] and len(csv)==1300
    assert csv.id.equals(test.id) and csv.id.is_unique and not csv.isna().any().any()
    assert set(csv.prediction.unique()).issubset({0,1})
    assert len(ruta.read_text().splitlines())==1301
    assert hashlib.sha256(ruta.read_bytes()).hexdigest()==r['sha256']
    with threadpool_limits(limits=1):
        s = scores(paquete['modelo'],test[COLUMNAS])
    assert np.isfinite(s).all() and np.array_equal(csv.prediction.to_numpy(),(s>=paquete['umbral']).astype(int))
    print('Submission correcto en estructura y reproducible con el modelo guardado.')
    print(csv.prediction.value_counts().sort_index().to_string())
    print('Las etiquetas ocultas no están disponibles; no se conoce el F1 del leaderboard.')

if __name__=='__main__': verificar()
