"""Semi-synthetic known-truth benchmark: real weather, synthetic power with a known PLR.

Why: injections into measured power test *sensitivity* to an added trend, but a normalisation model
that mis-describes the system's response (temperature coefficient, spectral or low-light behaviour)
can bias the rate when the weather mix changes between years -- and that bias cancels in an
injection difference. Here the truth is exact: power is generated from the measured POA and
temperature of real sites (real gaps, real inter-annual variability) with a known rate r.

Response families (the 'true' system):
  pvwatts    P = P_r * POA/1000 * (1 + g (Tc-25))                       -- exactly RdTools' model
  thin_film  P = P_r * POA/1000 * (1 + g_true (Tc-25)) * S(doy) * L(POA)
             g_true = 0.5 g_datasheet (thin films: weaker temperature loss than assumed),
             S = 1 + 0.03 cos(2 pi (doy - doy_summer)/365.25) (spectral gain in the warm season),
             L = 1 - 0.04 ln(1000/POA)^2 / ln(5)^2 clipped (low-light loss; zero at 1000 W/m2)
Both multiplied by (1 + r t) and 1 % Gaussian noise. RdTools always uses the datasheet g.
Truth: first-year-normalised rate = r exactly (no other trend in the generator).
Output: results/degradation_paper/semisynthetic_runs.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import ml_plr, perf_models as pm
from src.degradation.kappa_hydra_d import estimate
from src.degradation.rdtools_pipeline import cell_temperature, sensor_plr
from src.degradation.tier_a import load_system

OUT = ROOT / "results/degradation_paper/semisynthetic_runs.csv"
WEATHER = ["dkasc_7", "1423", "1430", "2107"]           # Alice Springs BWh, Henderson BWh, Golden Dfb, Arbuckle Csa
RATES = [-0.5, -1.0]
SEEDS = [0, 1, 2]
G_DATASHEET = -0.0040


def synthesize(frame: pd.DataFrame, meta: dict, family: str, rate: float, seed: int) -> pd.DataFrame:
    f = ml_plr.clean(frame)[["poa_wm2", "t_amb_c"] + (["t_module_c"] if "t_module_c" in frame else [])].copy()
    poa = f.poa_wm2.clip(lower=0)
    tc = cell_temperature(poa, f.get("t_amb_c"), f.get("t_module_c"))
    if family == "pvwatts":
        resp = 1 + G_DATASHEET * (tc - 25)
    else:
        summer = 355 if meta["lat"] < 0 else 172
        spectral = 1 + 0.03 * np.cos(2 * np.pi * (f.index.dayofyear - summer) / 365.25)
        lowlight = 1 - 0.04 * (np.log(1000 / poa.clip(lower=20)) ** 2) / np.log(5) ** 2
        resp = (1 + 0.5 * G_DATASHEET * (tc - 25)) * spectral * lowlight.clip(lower=0.5, upper=1.02)
    t = (f.index - f.index.min()).total_seconds() / (365.25 * 86400)
    rng = np.random.default_rng(seed)
    p = 5000.0 * poa / 1000 * resp * (1 + rate / 100 * t) * (1 + rng.normal(0, 0.01, len(f)))
    f["power_w"] = p.clip(lower=0).where(poa.notna())
    return f


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["weather", "family", "rate", "seed", "method"])
    keys = set(zip(done.weather, done.family, done.rate.astype(float), done.seed, done.method))
    for raw in WEATHER:
        frame, meta = load_system(raw)
        wid = meta["system_id"]
        for family in ("pvwatts", "thin_film"):
            for rate in RATES:
                for seed in SEEDS:
                    syn = synthesize(frame, meta, family, rate, seed)
                    rows = []
                    if (wid, family, rate, seed, "rdtools") not in keys:
                        r = sensor_plr(syn, 5.0, G_DATASHEET)
                        rows.append({"method": "rdtools", "plr": r["plr"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
                    d = None
                    for model in ("xgboost", "tcn"):
                        if (wid, family, rate, seed, model) in keys:
                            continue
                        d = d or ml_plr.prepare(syn, 5.0)
                        pred, _ = pm.fit_predict(model, hp[model]["params"], seed, (d["x"][d["train"]], d["y"][d["train"]]),
                                                 (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
                        r = ml_plr.plr_from_prediction(syn, 5.0, pred, d["stamps"])
                        rows.append({"method": model, "plr": r["plr"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
                    if (wid, family, rate, seed, "khd_full") not in keys:
                        r = estimate(syn, 5.0, G_DATASHEET, seeds=(seed,), jackknife=False, **hp["kappa_hydra_d"]["params"])
                        rows.append({"method": "khd_full", "plr": r["plr"], "ci_low": np.nan, "ci_high": np.nan,
                                     "coverage_80": r.get("coverage_80")})
                    for row in rows:
                        row.update({"weather": wid, "climate": meta["climate"],
                                    "family": family, "rate": rate, "seed": seed, "error": row["plr"] - rate})
                        print(row["weather"], family, rate, seed, row["method"], round(row["plr"], 3), flush=True)
                    if rows:
                        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
