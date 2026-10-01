"""RQ3 — SHAP drivers of the performance models, by climate group.

For every system with an RdTools reference PLR: XGBoost (HPO parameters, seeds 0-4) fitted on the
reference period; exact TreeSHAP on the held-out reference weeks. Features enter as the last three
hourly lags, so SHAP values are summed over lags per physical input. Output:
results/degradation_paper/shap_by_system.csv (mean |SHAP| share per input, averaged over seeds).
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

OUT = ROOT / "results/degradation_paper"
HOT_ARID = ("BWh", "BWk", "BSh", "BSk")


def main() -> None:
    import shap
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["xgboost"]["params"]
    ref = pd.read_csv(OUT / "reference_plr.csv")
    rows = []
    for sid, climate in ref[ref.status == "ok"][["system_id", "climate"]].itertuples(index=False):
        frame, meta = load_system(sid.replace("pvdaq_", ""))
        d = ml_plr.prepare(frame, meta["dc_kw"])
        train, val = (d["x"][d["train"]], d["y"][d["train"]]), (d["x"][d["val"]], d["y"][d["val"]])
        shares = []
        for seed in range(5):
            est = pm.fit_model("xgboost", hp, seed, train, val)
            values = shap.TreeExplainer(est).shap_values(pm._flat(val[0]))        # [n, 3 lags x features]
            per_input = np.abs(values).reshape(len(values), 3, len(pm.FEATURES)).sum(axis=1).mean(axis=0)
            shares.append(per_input / per_input.sum())
        share = np.mean(shares, axis=0)
        rows.append({"system_id": sid, "climate": climate, "group": "hot/arid" if climate in HOT_ARID else "other",
                     **{f"shap_{f}": float(v) for f, v in zip(pm.FEATURES, share)}})
        print(sid, {f: round(float(v), 3) for f, v in zip(pm.FEATURES, share)}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "shap_by_system.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
