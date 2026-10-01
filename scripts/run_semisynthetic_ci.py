"""Rate-interval coverage of kappa-HYDRA-D on the semi-synthetic records (exact truth).

The main semi-synthetic benchmark scored point estimates (and hourly quantile coverage); here the
95 % interval of the RATE is checked: same records (run_semisynthetic.synthesize), same interval as on
the measured records (five model seeds + leave-one-year-out jackknife). Coverage = share of records
whose interval contains the true r. The YoY bootstrap intervals of RdTools / XGBoost / TCN are already
stored in semisynthetic_runs.csv.
  python scripts/run_semisynthetic_ci.py [weather ids ...]      (parallel workers: RUN_TAG=_a ...)
Output: results/degradation_paper/semisynthetic_ci_runs{RUN_TAG}.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_semisynthetic import G_DATASHEET, RATES, SEEDS, WEATHER, synthesize  # noqa: E402
from src.degradation.kappa_hydra_d import estimate  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

COLUMNS = ["weather", "climate", "family", "rate", "noise_seed", "plr", "ci_low", "ci_high", "plr_se", "plr_seed_sd",
           "jackknife_years", "covered", "fit_seconds"]


def main(only: list[str]) -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    out = ROOT / "results/degradation_paper" / f"semisynthetic_ci_runs{os.environ.get('RUN_TAG', '')}.csv"
    parts = [pd.read_csv(p) for p in (ROOT / "results/degradation_paper").glob("semisynthetic_ci_runs*.csv")]
    done = pd.concat(parts) if parts else pd.DataFrame(columns=["weather", "family", "rate", "noise_seed"])
    keys = set(zip(done.weather, done.family, done.rate.astype(float), done.noise_seed))
    for raw in (only or WEATHER):
        frame, meta = load_system(raw)
        wid = meta["system_id"]
        for family in ("pvwatts", "thin_film"):
            for rate in RATES:
                for seed in SEEDS:
                    if (wid, family, rate, seed) in keys:
                        continue
                    syn = synthesize(frame, meta, family, rate, seed)
                    r = estimate(syn, 5.0, G_DATASHEET, seeds=(0, 1, 2, 3, 4), jackknife=True, **hp)
                    r.pop("_val", None)
                    row = {"weather": wid, "climate": meta["climate"], "family": family, "rate": rate, "noise_seed": seed,
                           **{k: r.get(k) for k in ("plr", "ci_low", "ci_high", "plr_se", "plr_seed_sd", "jackknife_years",
                                                    "fit_seconds")}}
                    row["covered"] = bool(row["ci_low"] <= rate <= row["ci_high"])
                    pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
                    print(wid, family, rate, seed, round(row["plr"], 3), [round(row["ci_low"], 3), round(row["ci_high"], 3)],
                          row["covered"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(0)
