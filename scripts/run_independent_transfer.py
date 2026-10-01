"""Geographically disjoint HPO/pretraining with later-period target tests.

Four leave-one-location-out folds, XGBoost and TCN fixed before this supplementary
experiment, 20 HPO trials per fold/model, five seeds. No target-location data enter
HPO or pretraining. Local six-month conditions use four months train plus two val.
The scratch_36 comparator uses 24 months train plus 12 val. All test after month 36.
This is a post-development sensitivity experiment, not preregistered validation.
"""
from __future__ import annotations
import sys,os,time,json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import run_independent_validation as iv
from src.degradation import perf_models as pm

OUT=iv.OUT/'geographic';OUT.mkdir(exist_ok=True)
MODELS=['xgboost','tcn'];CONDITIONS=['scratch_36','scratch_6','zero_shot','finetune_6']

def pool(parts,mask):
    return (np.concatenate([d['x'][d[mask]] for d in parts]),np.concatenate([d['y'][d[mask]] for d in parts]))

def objective_score(est,model,parts):
    rows=[]
    for d in parts:
        p=pm.predict(model,est,d['x'][d['val']])
        rows.append({'location':d['meta']['location'],'rmse':iv.scores(p,d['y'][d['val']])['rmse']})
    return float(pd.DataFrame(rows).groupby('location').rmse.mean().mean())

def main():
    all_data={s:iv.prepare(s) for s in iv.systems()}
    locations=sorted({d['meta']['location'] for d in all_data.values()})
    db='sqlite:///'+str((OUT/'hpo.sqlite3').resolve()).replace('\\','/')
    for loc in locations:
        parts=[d for d in all_data.values() if d['meta']['location']!=loc]
        targets=[(sid,d) for sid,d in all_data.items() if d['meta']['location']==loc]
        assert all(d['meta']['location']!=loc for d in parts)
        train,val=pool(parts,'train'),pool(parts,'val')
        for model in MODELS:
            hp_file=OUT/f'hpo_{loc}_{model}.json'
            if hp_file.exists():hp=json.loads(hp_file.read_text())
            else:
                study=optuna.create_study(study_name=f'{loc}_{model}',storage=db,load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(seed=42),direction='minimize')
                def objective(trial):
                    p=iv.search_params(trial,model)
                    est=pm.fit_model(model,p,0,train,val)
                    score=objective_score(est,model,parts)
                    print('GEO_HPO',loc,model,trial.number,score,flush=True);return score
                n=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
                study.optimize(objective,n_trials=max(0,20-n),gc_after_trial=True)
                hp={'params':iv.search_params(optuna.trial.FixedTrial(study.best_params),model),
                    'trials':20,'source_locations':sorted({d['meta']['location'] for d in parts}),
                    'excluded_location':loc,'val_rmse_location_mean':study.best_value}
                iv.save_json(hp_file,hp)
            for seed in iv.SEEDS:
                needed=[(sid,d) for sid,d in targets if any(not (OUT/f'{sid}_{model}_{seed}_{c}.json').exists() for c in CONDITIONS)]
                if not needed:continue
                t0=time.perf_counter();pre=pm.fit_model(model,hp['params'],seed,train,val);pre_seconds=time.perf_counter()-t0
                for sid,d in needed:
                    stamps=pd.to_datetime(d['stamps']);start=pd.Timestamp(d['meta']['start'])
                    four=start+pd.DateOffset(months=4);six=start+pd.DateOffset(months=6)
                    local_train=stamps<four
                    local_val=(stamps>=four+pd.Timedelta(hours=24))&(stamps<six)
                    assert local_train.sum()>50 and local_val.sum()>30,(sid,local_train.sum(),local_val.sum())
                    assert stamps[d['test']].min()>stamps[local_val].max()+pd.Timedelta(hours=23)
                    for cond in CONDITIONS:
                        row_file=OUT/f'{sid}_{model}_{seed}_{cond}.json'
                        if row_file.exists():continue
                        mask_train=d['train'] if cond=='scratch_36' else local_train
                        mask_val=d['val'] if cond=='scratch_36' else local_val
                        tr=(d['x'][mask_train],d['y'][mask_train]);va=(d['x'][mask_val],d['y'][mask_val])
                        begin=time.perf_counter()
                        if cond.startswith('scratch'):est=pm.fit_model(model,hp['params'],seed,tr,va)
                        elif cond=='zero_shot':est=pre
                        else:est=pm.finetune(model,pre,hp['params'],seed,tr,va)
                        fit_seconds=time.perf_counter()-begin
                        pred=pm.predict(model,est,d['x'][d['test']]);y=d['y'][d['test']]
                        assert np.isfinite(pred).all()
                        row={'system_id':sid,'location':loc,'model':model,'seed':seed,'condition':cond,
                             'n_local_train':0 if cond=='zero_shot' else int(mask_train.sum()),
                             'n_local_validation':0 if cond=='zero_shot' else int(mask_val.sum()),
                             'n_test':int(d['test'].sum()),'source_locations':hp['source_locations'],
                             'pretrain_seconds':pre_seconds,'local_fit_seconds':fit_seconds,**iv.scores(pred,y)}
                        np.savez_compressed(row_file.with_suffix('.npz'),stamps=d['stamps'][d['test']],y=y,pred=pred)
                        iv.save_json(row_file,row)
                        print('GEO_TEST',sid,model,seed,cond,row['rmse'],flush=True)
    rows=[json.loads(p.read_text()) for p in OUT.glob('*.json') if not p.name.startswith('hpo_')]
    assert len(rows)==600
    pd.DataFrame(rows).to_csv(OUT/'geographic_runs.csv',index=False)

if __name__=='__main__':
    # Permit an early independently launched worker; the supervisor can safely
    # reach this stage while that worker is still active without duplicate fits.
    import msvcrt
    with (OUT/'worker.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        while True:
            try:
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(30)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    sys.stdout.flush();os._exit(0)
