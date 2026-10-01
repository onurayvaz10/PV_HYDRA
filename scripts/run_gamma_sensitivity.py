"""Retrospective sensitivity to assumed temperature coefficient, same 15-array cohort."""
from pathlib import Path
import sys,os
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.degradation.tier_a import load_system
from src.degradation.ml_plr import clean
from src.degradation.rdtools_pipeline import gamma_for,sensor_plr
OUT=ROOT/'results/independent_validation_20260929'

def main():
    ref=pd.read_csv(ROOT/'results/degradation_paper/reference_plr.csv')
    file=OUT/'gamma_sensitivity.csv'
    rows=pd.read_csv(file).to_dict('records') if file.exists() else []
    done={(r['system_id'],float(r['factor'])) for r in rows}
    for sid in ref.loc[ref.status=='ok','system_id']:
        missing=[f for f in [.8,1.,1.2] if (sid,f) not in done]
        if not missing:continue
        # Match the original RdTools reference window exactly. ML's clean()
        # start-date screen would confound coefficient and record-window changes.
        frame,meta=load_system(sid.replace('pvdaq_',''))
        gamma=gamma_for(meta['label'])[0]
        for factor in missing:
            np.random.seed(20260929)
            result=sensor_plr(frame,meta['dc_kw'],gamma*factor)
            assert result['status']=='ok',(sid,factor,result)
            rows.append({'system_id':sid,'factor':factor,'gamma':gamma*factor,
                **{k:result[k] for k in ['plr','ci_low','ci_high','days','span_years']}})
            pd.DataFrame(rows).to_csv(file,index=False)
            print('GAMMA',sid,factor,result['plr'],flush=True)
    assert len(rows)==45

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
