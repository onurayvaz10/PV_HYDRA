"""Complete-result gate and calendar-aware, location-aware supplementary summaries."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.stats import holm
OUT=ROOT/'results/independent_validation_20260929'

def calendar_dm(a,b,stamps,lag=7):
    """Daily loss means; missing calendar days remain absent scores. Sandwich variance
    uses Bartlett weights on actual day offsets, denominator observed_days**2.
    No h=24 correction for contemporaneous power reconstruction. Conditional on the
    observation calendar; a DM-style equal-mean-loss diagnostic with asymptotic normal p.
    """
    diff=pd.Series(np.asarray(a)-np.asarray(b),index=pd.to_datetime(stamps)).resample('D').mean()
    n=int(diff.notna().sum());mean=float(diff.mean())
    u=(diff-mean).fillna(0).to_numpy()  # zero estimating-equation contribution, not an imputed error
    variance=float(u@u)
    for k in range(1,min(lag,len(u)-1)+1):variance+=2*(1-k/(lag+1))*float(u[k:]@u[:-k])
    variance/=n*n
    statistic=mean/np.sqrt(variance) if variance>0 else np.nan
    return {'dm_hac':statistic,'p_value':float(2*norm.sf(abs(statistic))),
            'n_observed_days':n,'calendar_days':len(diff),'mean_daily_loss_difference':mean,'hac_lag_days':lag}

def location_sign_test(differences):
    d=np.asarray(differences,dtype=float);observed=abs(d.mean())
    values=[abs((d*np.asarray(sign)).mean()) for sign in itertools.product([-1,1],repeat=len(d))]
    return float(np.mean(np.asarray(values)>=observed-1e-14))

def main():
    table=OUT/'tables';table.mkdir(exist_ok=True)
    r=pd.read_csv(OUT/'chronological_runs.csv')
    assert len(r)==600 and r.model.nunique()==8 and r.system_id.nunique()==15
    assert not r.duplicated(['system_id','model','seed']).any()
    assert r.groupby(['system_id','model']).seed.nunique().eq(5).all()
    assert np.isfinite(r[['rmse','mae','r2','train_seconds']]).all().all()
    seed_loc=r.groupby(['location','model','seed'])[['rmse','mae','r2']].mean().reset_index()
    seed_loc.groupby(['model','seed'])[['rmse','mae','r2']].mean().reset_index().to_csv(table/'chronological_equal_location_by_seed.csv',index=False)
    r.groupby(['system_id','location','model']).agg(rmse=('rmse','mean'),rmse_seed_sd=('rmse','std'),
        mae=('mae','mean'),r2=('r2','mean'),train_seconds=('train_seconds','mean')).reset_index().to_csv(table/'chronological_seed_summary.csv',index=False)
    common=[];tests=[];quantiles=[]
    for sid in sorted(r.system_id.unique()):
        predictions={};y=None;stamps=None
        for model in sorted(r.model.unique()):
            pred=[]
            for seed in range(5):
                with np.load(OUT/'predictions'/f'{sid}_{model}_{seed}.npz') as z:
                    if y is None:y=z['y'];stamps=z['stamps']
                    else:assert np.array_equal(stamps,z['stamps']) and np.array_equal(y,z['y'])
                    assert np.isfinite(z['pred']).all()
                    if model=='kappa_hydra_d':
                        lower,upper=z['q10'],z['q90']
                        quantiles.append({'system_id':sid,'location':r.loc[r.system_id==sid,'location'].iloc[0],
                          'seed':seed,'n':len(y),'coverage_80':float(np.mean((y>=lower)&(y<=upper))),
                          'mean_width_80':float(np.mean(upper-lower)),
                          'crossing_count':int(np.sum(lower>upper)),
                          'negative_median_count':int(np.sum(z['pred']<0))})
                    pred.append(z['pred'])
            predictions[model]=np.mean(pred,axis=0)
            err=predictions[model]-y
            common.append({'system_id':sid,'location':r.loc[r.system_id==sid,'location'].iloc[0],'model':model,
               'n':len(y),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(abs(err))),
               'r2':float(1-np.sum(err**2)/np.sum((y-y.mean())**2))})
        for model,pred in predictions.items():
            if model=='xgboost':continue
            result=calendar_dm((pred-y)**2,(predictions['xgboost']-y)**2,stamps)
            tests.append({'system_id':sid,'model':model,'comparator':'xgboost',**result})
    common=pd.DataFrame(common);common.to_csv(table/'chronological_ensemble_by_array.csv',index=False)
    pd.DataFrame(quantiles).to_csv(table/'chronological_quantile_diagnostics.csv',index=False)
    loc=common.groupby(['location','model'])[['rmse','mae','r2']].mean().reset_index()
    loc.to_csv(table/'chronological_by_location.csv',index=False)
    loc.groupby('model')[['rmse','mae','r2']].agg(['mean','std']).to_csv(table/'chronological_equal_location.csv')
    common.groupby('model')[['rmse','mae','r2']].agg(['mean','std']).to_csv(table/'chronological_equal_array.csv')
    test=pd.DataFrame(tests)
    adjusted=holm({str(i):p for i,p in enumerate(test.p_value)})
    test['p_holm_all_array_model_tests']=[adjusted.get(str(i),np.nan) for i in range(len(test))]
    test.to_csv(table/'calendar_dm_hac.csv',index=False)
    wide=loc.pivot(index='location',columns='model',values='rmse')
    pd.DataFrame([{'model':m,'comparator':'xgboost','locations':len(wide),
      'mean_location_rmse_difference':float((wide[m]-wide.xgboost).mean()),
      'exact_sign_permutation_p':location_sign_test(wide[m]-wide.xgboost)} for m in wide if m!='xgboost']).to_csv(table/'location_sign_tests.csv',index=False)
    geo=pd.read_csv(OUT/'geographic/geographic_runs.csv')
    assert len(geo)==600 and not geo.duplicated(['system_id','model','seed','condition']).any()
    assert geo.groupby(['system_id','model','condition']).seed.nunique().eq(5).all()
    assert geo.system_id.nunique()==15 and geo.model.nunique()==2 and geo.condition.nunique()==4
    assert np.isfinite(geo[['rmse','mae','r2']]).all().all()
    geo_seed=geo.groupby(['location','model','condition','seed'])[['rmse','mae','r2']].mean().reset_index()
    geo_seed.groupby(['model','condition','seed'])[['rmse','mae','r2']].mean().reset_index().to_csv(table/'geographic_equal_location_by_seed.csv',index=False)
    geo.groupby(['system_id','location','model','condition']).agg(rmse=('rmse','mean'),rmse_seed_sd=('rmse','std'),
        mae=('mae','mean'),r2=('r2','mean')).reset_index().to_csv(table/'geographic_by_array.csv',index=False)
    gl=geo.groupby(['location','model','condition'])[['rmse','mae','r2']].mean().reset_index()
    gl.to_csv(table/'geographic_by_location.csv',index=False)
    gl.groupby(['model','condition'])[['rmse','mae','r2']].mean().to_csv(table/'geographic_equal_location.csv')
    # Connect geographic transfer to the paper's rate estimand on the later test
    # period. This is agreement with a same-sample physical reference, NOT field truth.
    from src.degradation.rdtools_pipeline import yoy_from_normalized
    rate_rows=[]
    for sid in sorted(geo.system_id.unique()):
        with np.load(OUT/'cache'/f'{sid}.npz') as z:
            mask=z['test'];stamps=z['stamps'][mask];truth=z['y'][mask];features=z['x'][mask,-1,:];physical=z['scale'][mask]
        meta=json.loads((OUT/'cache'/f'{sid}.json').read_text())
        index=pd.to_datetime(stamps)
        poa=pd.Series(features[:,0]*1000,index=index)
        temperature=pd.Series(features[:,1]*60,index=index)
        power=pd.Series(truth*meta['dc_kw']*1000,index=index)
        np.random.seed(20260929)
        reference=yoy_from_normalized(pd.Series(truth/physical,index=index),poa,temperature,power,False,60)
        rate_rows.append({'system_id':sid,'location':meta['location'],'model':'rdtools','condition':'same_test_samples',**reference})
        for model in ('xgboost','tcn'):
            for condition in ('scratch_36','scratch_6','zero_shot','finetune_6'):
                predictions=[]
                for seed in range(5):
                    with np.load(OUT/'geographic'/f'{sid}_{model}_{seed}_{condition}.npz') as z:
                        assert np.array_equal(z['stamps'],stamps) and np.array_equal(z['y'],truth)
                        predictions.append(z['pred'])
                pred=np.mean(predictions,axis=0)
                normalized=pd.Series(np.divide(truth,pred,out=np.full(len(pred),np.nan,dtype=float),where=pred>.02),index=index)
                np.random.seed(20260929)
                rate=yoy_from_normalized(normalized,poa,temperature,power,False,60)
                rate_rows.append({'system_id':sid,'location':meta['location'],'model':model,'condition':condition,
                    'abs_difference_reference':abs(rate.get('plr',np.nan)-reference.get('plr',np.nan)),**rate})
    rates=pd.DataFrame(rate_rows)
    rates.to_csv(table/'geographic_test_period_plr.csv',index=False)
    assert len(rates)==135 and rates.status.eq('ok').all() and np.isfinite(rates.plr).all()
    # Rate coverage is reported against both conventions, not silently conflated.
    old=ROOT/'results/degradation_paper'
    ci=pd.concat([pd.read_csv(p) for p in old.glob('semisynthetic_ci_runs*.csv')],ignore_index=True)
    keys=['weather','family','rate','noise_seed']
    assert not ci.duplicated(keys).any() and len(ci)==48
    assert ci[['ci_low','ci_high','plr']].notna().all().all()
    ci['method']='khd_full'
    other=pd.read_csv(old/'semisynthetic_runs.csv');other=other[other.method!='khd_full'].rename(columns={'seed':'noise_seed'})
    all_ci=pd.concat([ci,other],ignore_index=True)
    all_ci['nominal_covered']=(all_ci.ci_low<=all_ci.rate)&(all_ci.ci_high>=all_ci.rate)
    all_ci['first_year_generator_rate']=all_ci.rate/(1+.5*all_ci.rate/100)
    all_ci['first_year_covered']=(all_ci.ci_low<=all_ci.first_year_generator_rate)&(all_ci.ci_high>=all_ci.first_year_generator_rate)
    all_ci['interval_width']=all_ci.ci_high-all_ci.ci_low
    all_ci.to_csv(table/'rate_coverage_by_record.csv',index=False)
    all_ci.groupby(['family','method']).agg(records=('rate','size'),nominal_coverage=('nominal_covered','mean'),
        first_year_convention_coverage=('first_year_covered','mean'),mean_width=('interval_width','mean')).to_csv(table/'rate_coverage_summary.csv')
    gamma=pd.read_csv(OUT/'gamma_sensitivity.csv')
    assert len(gamma)==45 and gamma.groupby('system_id').factor.nunique().eq(3).all()
    assert np.isfinite(gamma.plr).all()
    gamma.groupby('system_id').agg(plr_min=('plr','min'),plr_max=('plr','max'),
        plr_range=('plr',lambda s:s.max()-s.min())).to_csv(table/'temperature_coefficient_sensitivity.csv')
    (OUT/'EXPERIMENTS_COMPLETE.json').write_text(json.dumps({'chronological_runs':600,'geographic_runs':600,
      'rate_interval_records':48,'gamma_sensitivity_runs':45,'status':'computed; manuscript integration and scientific review still required'},indent=2))
    print('COMPLETE 600 chronological + 600 geographic + 48 rate CI records',flush=True)

if __name__=='__main__':main()
