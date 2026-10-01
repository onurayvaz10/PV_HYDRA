"""Reference-window sensitivity of two-stage ML PLR (T12).

The two-stage method needs a reference period for its performance model; RdTools (physics model)
and kappa-HYDRA-D (whole-record fit) do not. If degradation were a uniform multiplicative factor,
the two-stage PLR would not depend on which 24 data months are used. Here the window is shifted:
data months 1-24 (standard), 13-36 and 25-48; XGBoost with the tuned parameters, 5 seeds.
Output: results/degradation_paper/tables/T12_refwindow_sensitivity.csv
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
from src.degradation.tier_a import load_system

WINDOWS = (0, 12, 24)
SEEDS = range(5)


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["xgboost"]["params"]
    ref = pd.read_csv(ROOT / "results/degradation_paper/reference_plr.csv")
    ref = ref[ref.status == "ok"].set_index("system_id")
    rows = []
    for sid in ref.index:
        frame, meta = load_system(sid.replace("pvdaq_", ""))
        d = ml_plr.prepare(frame, meta["dc_kw"])
        y = pd.Series(d["y"], index=d["stamps"])
        counts = y.notna().groupby(y.index.to_period("M")).sum()
        good = counts[counts >= 150].index
        week5 = d["stamps"].isocalendar().week.to_numpy() % 5 == 0
        row = {"system_id": sid, "label": ref.loc[sid, "label"], "climate": ref.loc[sid, "climate"],
               "rdtools_plr": ref.loc[sid, "plr"]}
        for first in WINDOWS:
            if first + 24 > len(good):
                row[f"plr_m{first + 1}_{first + 24}"] = np.nan
                continue
            start, end = good[first].to_timestamp(), good[first + 23].to_timestamp(how="end")
            inref = (d["stamps"] >= start) & (d["stamps"] <= end) & np.isfinite(d["y"])
            tr, va = inref & ~week5, inref & week5
            plrs = []
            for seed in SEEDS:
                pred, _ = pm.fit_predict("xgboost", hp, seed, (d["x"][tr], d["y"][tr]), (d["x"][va], d["y"][va]), d["x"])
                plrs.append(ml_plr.plr_from_prediction(frame, meta["dc_kw"], pred, d["stamps"])["plr"])
            row[f"plr_m{first + 1}_{first + 24}"] = float(np.mean(plrs))
            row[f"sd_m{first + 1}_{first + 24}"] = float(np.std(plrs, ddof=1))
        vals = [row[k] for k in row if k.startswith("plr_m") and np.isfinite(row[k])]
        row["window_range"] = float(max(vals) - min(vals)) if len(vals) > 1 else np.nan
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "label"}, flush=True)
    out = ROOT / "results/degradation_paper/tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "T12_refwindow_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
