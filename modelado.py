from pathlib import Path
from time import perf_counter
import json
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
SEMILLA=42
COLUMNAS=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']

class VariablesVino(BaseEstimator, TransformerMixin):
    def __init__(self, ratios=False): self.ratios=ratios
    def fit(self,X,y=None): return self
    def transform(self,X):
        X=pd.DataFrame(X,columns=COLUMNAS).copy()
        if self.ratios:
            X['ratio_acidez']=X.fixed_acidity/X.volatile_acidity.replace(0,np.nan)
            X['ratio_so2']=X.free_sulfur_dioxide/X.total_sulfur_dioxide.replace(0,np.nan)
            X['ratio_alcohol_densidad']=X.alcohol/X.density.replace(0,np.nan)
        return X.replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)

def cargar():
    a=pd.read_csv(ROOT/'datos_estudiantes/train_public.csv'); b=pd.read_csv(ROOT/'datos_estudiantes/test_public_features.csv')
    assert len(a)==5197 and len(b)==1300 and a.id.is_unique and b.id.is_unique and not set(a.id)&set(b.id)
    assert set(a.target.unique())=={0,1} and 'target' not in b
    return a,b

def pipeline(modelo,ratios=False,pca=None,robust=False):
    return Pipeline([('variables',VariablesVino(ratios)),('imputar',SimpleImputer(strategy='median')),('escalar',RobustScaler() if robust else StandardScaler()),('pca',PCA(n_components=pca,svd_solver='full') if pca else 'passthrough'),('modelo',modelo)])

def ajustar(m,X,y,b):
    return m.fit(X,y,**({'modelo__sample_weight':compute_sample_weight('balanced',y)} if b else {}))
def scores(m,X):
    return m.decision_function(X) if hasattr(m,'decision_function') else m.predict_proba(X)[:,1]
def metricas(y,p,s):
    return dict(f1_macro=f1_score(y,p,average='macro'),accuracy=accuracy_score(y,p),precision_1=precision_score(y,p,zero_division=0),recall_1=recall_score(y,p,zero_division=0),f1_1=f1_score(y,p,zero_division=0),average_precision=average_precision_score(y,s))
def elegir_umbral(y,s,d):
    grid=np.linspace(.12,.78,67) if d==.5 else np.quantile(s,np.linspace(.65,.95,61))
    return float(max(np.unique(np.r_[grid,d]),key=lambda u:(f1_score(y,s>=u,average='macro'),-abs(u-d))))


def pesos_potencia(d,p):
    ceros=d==0
    w=np.divide(1.,d**p,out=np.zeros_like(d),where=~ceros)
    filas=ceros.any(axis=1); w[filas]=ceros[filas].astype(float)
    return w

def peso2(d): return pesos_potencia(d,2)
def peso3(d): return pesos_potencia(d,3)

def crear(c):
    f=c['familia']
    if f=='k-NN':
        m=KNeighborsClassifier(n_neighbors=c['k'],metric=c['distancia'],weights={1:'distance',2:peso2,3:peso3}[c['potencia']],n_jobs=1)
    elif f=='SVM':
        m=SVC(C=c['C'],gamma=c['gamma'],kernel=c['kernel'],degree=2,class_weight='balanced' if c['balance'] else None,cache_size=64,max_iter=40000)
    else:
        m=MLPClassifier(hidden_layer_sizes=tuple(c['capas']),activation=c['act'],alpha=c['alpha'],batch_size=64,learning_rate_init=c['lr'],max_iter=220,early_stopping=True,validation_fraction=.15,n_iter_no_change=18,random_state=SEMILLA)
    return pipeline(m,c['ratios'],c['pca'] or None,c['robust'])

def nombre(c):
    f=c['familia']; b=f"ratios={int(c['ratios'])} robust={int(c['robust'])} PCA={c['pca']}"
    if f=='k-NN': return f"k-NN k={c['k']} {c['distancia']} 1/d^{c['potencia']} {b}"
    if f=='SVM': return f"SVM {c['kernel']} C={c['C']} gamma={c['gamma']} balance={c['balance']} {b}"
    return f"MLP {c['capas']} {c['act']} alpha={c['alpha']} lr={c['lr']} balance={c['balance']} {b}"

def umbral_rapido(y,s,d):
    grid=np.unique(np.r_[d,np.arange(.12,.801,.01)]) if d==.5 else np.unique(np.r_[0.,np.quantile(s,np.linspace(.60,.98,77))])
    p=s[:,None]>=grid[None,:]; tp=(p & (y[:,None]==1)).sum(0); fp=(p & (y[:,None]==0)).sum(0); fn=(y==1).sum()-tp; tn=(y==0).sum()-fp
    f=(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))/2
    return float(grid[np.lexsort((np.abs(grid-d),-f))[0]])

def evaluar(c,X,y,folds):
    s=np.empty(len(y)); regs=[]
    for i,(tr,va) in enumerate(folds,1):
        m=crear(c)
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always',ConvergenceWarning)
            t=perf_counter(); ajustar(m,X.iloc[tr],y[tr],c['familia']=='MLP' and c['balance']); fit=perf_counter()-t
        t=perf_counter(); s[va]=scores(m,X.iloc[va]); pred=perf_counter()-t
        regs.append(dict(fold=i,ajuste_s=fit,inferencia_s=pred,iteraciones=int(np.max(getattr(m.named_steps['modelo'],'n_iter_',0))),alerta_convergencia=any(issubclass(a.category,ConvergenceWarning) for a in avisos)))
    return s,regs

def buscar(X,y,folds,grupos,salida):
    inicio=perf_counter(); filas=[]; detalle=[]; cache=[]; configs={}; puntuaciones={}
    def registrar(c,s,regs,origen):
        n=nombre(c); d=0. if c['familia']=='SVM' else .5
        t=perf_counter(); u=umbral_rapido(y,s,d); tu=perf_counter()-t
        met=metricas(y,s>=u,s); mf=[]
        for (tr,va),r in zip(folds,regs):
            # F1 por fold con umbral global de selección OOF; no es una nueva validación.
            f=f1_score(y[va],s[va]>=u,average='macro'); mf.append(f)
            detalle.append(dict(modelo=n,familia=c['familia'],origen=origen,umbral=u,**r,f1_macro=f))
        filas.append(dict(modelo=n,familia=c['familia'],origen=origen,umbral=u,f1_default=f1_score(y,s>=d,average='macro'),f1_media_folds=np.mean(mf),f1_std_folds=np.std(mf),ajuste_medio_s=np.mean([r['ajuste_s'] for r in regs]),inferencia_media_s=np.mean([r['inferencia_s'] for r in regs]),tiempo_umbral_s=tu,alertas_convergencia=sum(r.get('alerta_convergencia',False) for r in regs),**met))
        configs[n]=c; puntuaciones[n]=s
    tiempos={}
    with threadpool_limits(limits=1):
        t0=perf_counter(); ks=[3,5,7,9,15,25,35,50,75,100]
        for ratios in [False,True]:
            for robust in [False,True]:
                for pca in [0,.95]:
                    for dist in ['manhattan','euclidean']:
                        base=dict(familia='k-NN',ratios=ratios,robust=robust,pca=pca,distancia=dist); vecinos=[]
                        for i,(tr,va) in enumerate(folds,1):
                            prep=pipeline(KNeighborsClassifier(),ratios,pca or None,robust)[:-1]
                            t=perf_counter(); z=prep.fit_transform(X.iloc[tr]); nn=NearestNeighbors(n_neighbors=max(ks),metric=dist,n_jobs=1).fit(z); fit=perf_counter()-t
                            t=perf_counter(); d,idx=nn.kneighbors(prep.transform(X.iloc[va])); pred=perf_counter()-t
                            vecinos.append((d,y[tr][idx])); cache.append(dict(**base,fold=i,ajuste_compartido_s=fit,inferencia_compartida_s=pred))
                        for k in ks:
                            for potencia in [1,2,3]:
                                c=dict(**base,k=k,potencia=potencia); s=np.empty(len(y)); regs=[]
                                for i,((tr,va),(d,et)) in enumerate(zip(folds,vecinos),1):
                                    t=perf_counter(); w=pesos_potencia(d[:,:k],potencia); s[va]=(w*et[:,:k]).sum(1)/w.sum(1); dt=perf_counter()-t
                                    regs.append(dict(fold=i,ajuste_s=0.,inferencia_s=dt,iteraciones=0,alerta_convergencia=False))
                                registrar(c,s,regs,'cache_vecinos')
        tiempos['busqueda_knn']=perf_counter()-t0; t0=perf_counter()
        candidatos=[]
        for ratios,pca in [(True,0),(False,0),(True,.95)]:
            for C in [3,10,50]:
                for gamma in ['scale',.3]:
                    for bal in [False,True]: candidatos.append(dict(familia='SVM',ratios=ratios,robust=False,pca=pca,kernel='rbf',C=C,gamma=gamma,balance=bal))
        for kernel in ['linear','poly']:
            for C in [.3,3]:
                for bal in [False,True]: candidatos.append(dict(familia='SVM',ratios=True,robust=False,pca=0,kernel=kernel,C=C,gamma='scale',balance=bal))
        for c in candidatos:
            s,regs=evaluar(c,X,y,folds); registrar(c,s,regs,'pipeline_directo')
        tiempos['busqueda_svm']=perf_counter()-t0; t0=perf_counter()
        for capas,act,alpha,lr,bal,ratios,pca in [([32,16],'relu',.001,.001,False,True,0),([64,32],'relu',.001,.001,True,True,0),([64,32],'tanh',.01,.001,True,True,0),([128,64],'relu',.0001,.001,False,True,0),([64,32],'relu',.001,.003,True,True,0),([64,32],'relu',.01,.001,False,False,0),([64,32],'tanh',.001,.003,False,True,0),([64,32],'relu',.001,.001,True,True,.95),([64,32],'tanh',.01,.001,False,False,.95)]:
            c=dict(familia='MLP',ratios=ratios,robust=False,pca=pca,capas=capas,act=act,alpha=alpha,lr=lr,balance=bal); s,regs=evaluar(c,X,y,folds); registrar(c,s,regs,'pipeline_directo')
        tiempos['busqueda_mlp']=perf_counter()-t0; t0=perf_counter()
        # Repetir ocho mejores k-NN con el pipeline exacto (empates en frontera k).
        top=sorted([r for r in filas if r['familia']=='k-NN'],key=lambda r:r['f1_macro'],reverse=True)[:8]
        for r in top:
            n=r['modelo']; c=configs[n]; s,regs=evaluar(c,X,y,folds); filas.remove(r); detalle[:]=[x for x in detalle if x['modelo']!=n]; registrar(c,s,regs,'pipeline_verificado')
        tiempos['verificacion_knn']=perf_counter()-t0
        tabla=pd.DataFrame(filas); tabla['elegible']=(tabla.origen!='cache_vecinos') & (tabla.alertas_convergencia==0); tabla=tabla.sort_values('f1_macro',ascending=False).reset_index(drop=True)
        ganador=tabla[tabla.elegible].iloc[0]; c=configs[ganador.modelo]; u=float(ganador.umbral)
        sg=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEMILLA).split(X,y,grupos)); assert all(not set(grupos[tr]) & set(grupos[va]) for tr,va in sg)
        t=perf_counter(); sg_s,_=evaluar(c,X,y,sg); audit=metricas(y,sg_s>=u,sg_s); tiempos['auditoria_grupos']=perf_counter()-t
    tabla.to_csv(salida/'avanzado_metricas.csv',index=False); pd.DataFrame(detalle).to_csv(salida/'avanzado_folds.csv',index=False); pd.DataFrame(cache).to_csv(salida/'avanzado_cache_vecinos.csv',index=False)
    (salida/'avanzado_configuraciones.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
    tiempos['total']=perf_counter()-inicio
    resumen=dict(configuraciones=len(tabla),n_knn=int((tabla.familia=='k-NN').sum()),n_svm=int((tabla.familia=='SVM').sum()),n_mlp=int((tabla.familia=='MLP').sum()),ajustes_reales=len(cache)+3*(int((tabla.familia!='k-NN').sum())+8),ajustes_auditoria=3,ganador=ganador.to_dict(),configuracion=c,matriz_confusion_oof=confusion_matrix(y,puntuaciones[ganador.modelo]>=u,labels=[0,1]).tolist(),auditoria_grupos=audit,tiempos_s=tiempos)
    (salida/'resumen_avanzado.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    return tabla,resumen
