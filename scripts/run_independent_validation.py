"""Post-development chronological validation. Never changes the retrospective experiment.

24 calendar months train, next 12 validate, remaining record test; 24 h embargo.
Only observed covariates, no interpolation. All methods share the same samples.
20 new HPO trials per model on the three named development arrays, five final seeds.
Contemporaneous weather is available: this tests later-period power reconstruction,
not an operational ahead-of-time weather/power forecast. Result selection is post-development.
"""
from __future__ import annotations
import os, sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import optuna

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.degradation import perf_models as pm, kappa_hydra_d as kd
from src.degradation.ml_plr import clean, scores
from src.degradation.rdtools_pipeline import gamma_for
from src.degradation.tier_a import load_system
from run_ml_plr import space

OUT=ROOT/'results/independent_validation_20260929'
OUT.mkdir(exist_ok=True,parents=True)
MODELS=list(pm.ALL_MODELS)+['kappa_hydra_d']
DEV=['dkasc_7','dkasc_12','dkasc_14']
SEEDS=list(range(5));TRIALS=20
torch.set_num_threads(2)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def save_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def systems():
    r=pd.read_csv(ROOT/'results/degradation_paper/reference_plr.csv')
    return list(r.loc[r.status=='ok','system_id'])

def prepare(sid):
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    file=cache/f'{sid}.npz';meta_file=cache/f'{sid}.json'
    if file.exists() and meta_file.exists():
        with np.load(file) as z:d={k:z[k] for k in z.files}
        d['meta']=json.loads(meta_file.read_text(encoding='utf-8'));return d
    frame,meta=load_system(sid.replace('pvdaq_',''))
    frame=clean(frame)
    # Determine cell-temperature source from the training period only. The helper
    # otherwise chooses based on missingness across the whole archive.
    start=frame.index.min();tr_end=start+pd.DateOffset(months=24);va_end=tr_end+pd.DateOffset(months=12)
    module=('t_module_c' in frame and frame.loc[frame.index<tr_end,'t_module_c'].notna().mean()>.5)
    if not module:frame=frame.drop(columns=['t_module_c'],errors='ignore')
    # Build directly to avoid any whole-record sensor-availability decision.
    from src.degradation.rdtools_pipeline import cell_temperature
    poa=frame.poa_wm2.clip(lower=0)
    if module:tcell=frame.t_module_c+3*poa/1000
    else:tcell=cell_temperature(poa,frame.get('t_amb_c'),None)
    tamb=frame.t_amb_c if 't_amb_c' in frame else tcell
    idx=frame.index;hour=idx.hour+.5
    features=pd.DataFrame({'poa_n':poa/1000,'tcell_n':tcell/60,'tamb_n':tamb.fillna(tcell)/40,
       'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
       'doy_sin':np.sin(2*np.pi*idx.dayofyear/365.25),'doy_cos':np.cos(2*np.pi*idx.dayofyear/365.25)},index=idx)
    target=frame.power_w/(meta['dc_kw']*1000)
    x,y,stamps=pm.windows(features,target)
    f=x[:,-1,:];ratio=y/np.maximum(f[:,0],1e-9)
    # Fixed pointwise plausibility filters only; no test-fitted quantile clipping,
    # no centered outage filter. Test anomalies outside these bounds are disclosed.
    ok=np.isfinite(y)&(f[:,0]>=.2)&(f[:,0]<=1.2)&(f[:,1]*60>=-50)&(f[:,1]*60<=110)&(y>=0)&(y<=1.1)&(ratio>=.01)&(ratio<=2)
    x,y,stamps=x[ok],y[ok],stamps[ok]
    tr=stamps<tr_end
    va=(stamps>=tr_end+pd.Timedelta(hours=24))&(stamps<va_end)
    te=stamps>=va_end+pd.Timedelta(hours=24)
    assert tr.sum()>500 and va.sum()>100 and te.sum()>500,(sid,tr.sum(),va.sum(),te.sum())
    assert stamps[tr].max()<stamps[va].min()-pd.Timedelta(hours=23)
    assert stamps[va].max()<stamps[te].min()-pd.Timedelta(hours=23)
    gamma=gamma_for(meta['label'])[0]
    year=((stamps-start).total_seconds()/(365.25*86400)).to_numpy(np.float32)
    mid=float((year[tr].min()+year[tr].max())/2)
    scale=x[:,-1,0]*(1+gamma*(x[:,-1,1]*60-25))
    assert (scale>0).all()
    d={'x':x,'y':y,'stamps':stamps.to_numpy(dtype='datetime64[ns]').astype('int64'),
       'train':tr,'val':va,'test':te,'years':year-mid,'scale':scale.astype(np.float32)}
    meta.update(start=str(start),train_end=str(tr_end),validation_end=str(va_end),
        test_end=str(stamps.max()),n_train=int(tr.sum()),n_val=int(va.sum()),n_test=int(te.sum()),
        location='Alice_Springs' if sid.startswith('dkasc') else sid,
        module_sensor_selected_on_train=bool(module),gamma=gamma,time_center=mid,
        feature_interpolation=False,split_embargo_hours=24)
    np.savez_compressed(file,**d);save_json(meta_file,meta)
    d['meta']=meta
    print('PREPARED',sid,meta['n_train'],meta['n_val'],meta['n_test'],flush=True)
    return d

def design(d,mask):
    f=d['x'][mask,-1,:]
    data=pd.DataFrame(f,columns=kd.FEATURES)
    data['poa']=f[:,0]*1000
    data['scale']=d['scale'][mask]
    data['kappa']=d['y'][mask]/d['scale'][mask]
    data['t_years']=d['years'][mask]
    data['val']=False
    return data

def fit(model,params,seed,d,apply_mask):
    started=time.perf_counter()
    if model=='kappa_hydra_d':
        net,rate=kd._fit(design(d,d['train']),seed,params['hidden'],params['lr'],params['steps'],'cuda')
        duration=time.perf_counter()-started
        pred=kd._predict(net,design(d,apply_mask),'cuda')*d['scale'][apply_mask,None]
        return pred[:,1],{'train_seconds':duration,'parameters':sum(p.numel() for p in net.parameters()),
                         'fitted_rate_mid_pct':rate*100,'q10':pred[:,0],'q90':pred[:,2]}
    pred,info=pm.fit_predict(model,params,seed,(d['x'][d['train']],d['y'][d['train']]),
                            (d['x'][d['val']],d['y'][d['val']]),d['x'][apply_mask])
    return pred,info

def search_params(trial,model):
    if model=='kappa_hydra_d':
        return {'hidden':trial.suggest_categorical('hidden',[32,64,128]),
                'lr':trial.suggest_float('lr',1e-3,1e-2,log=True),
                'steps':trial.suggest_categorical('steps',[2000,3000,5000])}
    return space(trial,model)

def hpo(models):
    db='sqlite:///'+str((OUT/'hpo.sqlite3').resolve()).replace('\\','/')
    data={sid:prepare(sid) for sid in DEV}
    hp_file=OUT/'hyperparameters.json'
    hp=json.loads(hp_file.read_text()) if hp_file.exists() else {}
    for model in models:
        if model in hp:continue
        study=optuna.create_study(study_name=model,storage=db,load_if_exists=True,direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
        def objective(trial):
            p=search_params(trial,model);values=[]
            for d in data.values():
                pred,_=fit(model,p,0,d,d['val']);values.append(scores(pred,d['y'][d['val']])['rmse'])
            value=float(np.mean(values));print('HPO',model,trial.number,value,flush=True);return value
        n=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,TRIALS-n),gc_after_trial=True)
        assert sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)>=TRIALS
        hp[model]={'params':search_params(optuna.trial.FixedTrial(study.best_params),model),
                   'val_rmse':study.best_value,'trials':TRIALS,'development_systems':DEV,
                   'training_period':'first 24 calendar months','validation_period':'next 12 calendar months'}
        save_json(hp_file,hp)
        study.trials_dataframe().to_csv(OUT/f'hpo_{model}.csv',index=False)

def run(models):
    hp=json.loads((OUT/'hyperparameters.json').read_text())
    dest=OUT/'predictions';dest.mkdir(exist_ok=True)
    rows_file=OUT/'chronological_runs.csv'
    for sid in systems():
        d=prepare(sid)
        for model in models:
            for seed in SEEDS:
                file=dest/f'{sid}_{model}_{seed}.npz';row_file=dest/f'{sid}_{model}_{seed}.json'
                if file.exists() and row_file.exists():continue
                pred,info=fit(model,hp[model]['params'],seed,d,d['test'])
                arrays={k:info.pop(k) for k in ('q10','q90') if k in info}
                assert np.isfinite(pred).all()
                row={'system_id':sid,'location':d['meta']['location'],'model':model,'seed':seed,
                     'n_train':int(d['train'].sum()),'n_val':int(d['val'].sum()),'n_test':int(d['test'].sum()),
                     **scores(pred,d['y'][d['test']]),**info}
                if 'q10' in arrays:
                    y=d['y'][d['test']];row['coverage_80']=float(((y>=arrays['q10'])&(y<=arrays['q90'])).mean())
                np.savez_compressed(file,stamps=d['stamps'][d['test']],y=d['y'][d['test']],pred=pred,**arrays)
                save_json(row_file,row)
                pd.DataFrame([json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]).to_csv(rows_file,index=False)
                print('TEST',sid,model,seed,row['rmse'],flush=True)

def checks():
    manifest=[]
    for sid in systems():
        d=prepare(sid)
        assert not np.any(d['train']&d['val']) and not np.any(d['val']&d['test'])
        assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
        manifest.append(d['meta'])
    save_json(OUT/'split_manifest.json',manifest)
    print('SPLIT_CHECKS_PASSED',len(manifest),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['check','hpo','run','all']);ap.add_argument('--models',nargs='+',default=MODELS)
    args=ap.parse_args()
    if args.mode in ('all','check'):checks()
    if args.mode in ('all','hpo'):hpo(args.models)
    if args.mode in ('all','run'):run(args.models)
    sys.stdout.flush();os._exit(0)
