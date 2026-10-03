"""Reference-window sensitivity of the two-stage rate on the 18 main systems with the fixed hour mask (same records,
target hygiene and mask as run_fixed_mask.py): XGBoost trained on data months 1-24, 13-36 and 25-48, 5 seeds.
Output: results/fixed_mask/refwindow_fixed.csv
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
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_mask as fm  # noqa: E402
from src.degradation import ml_plr, perf_models as pm  # noqa: E402

WINDOWS = (0, 12, 24)


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["xgboost"]["params"]
    rows = []
    for sid in fm.systems():
        base, meta, gamma, ym, hh = fm.load(sid)
        d = ml_plr.prepare(base, meta["dc_kw"], healthy=hh, full_record=True)
        y = pd.Series(d["y"], index=d["stamps"])
        counts = y.notna().groupby(y.index.to_period("M")).sum()
        good = counts[counts >= 150].index
        week5 = d["stamps"].isocalendar().week.to_numpy() % 5 == 0
        row = {"system_id": sid}
        for first in WINDOWS:
            key = f"plr_m{first + 1}_{first + 24}"
            if first + 24 > len(good):
                row[key] = np.nan
                continue
            start, end = good[first].to_timestamp(), good[first + 23].to_timestamp(how="end")
            inref = (d["stamps"] >= start) & (d["stamps"] <= end) & np.isfinite(d["y"])
            tr, va = inref & ~week5, inref & week5
            plrs = []
            for seed in fm.SEEDS:
                pred, _ = pm.fit_predict("xgboost", hp, seed, (d["x"][tr], d["y"][tr]), (d["x"][va], d["y"][va]), d["x"])
                plrs.append(ml_plr.plr_from_prediction(base, meta["dc_kw"], pred, d["stamps"], mask=ym)["plr"])
            row[key] = float(np.mean(plrs))
        vals = [v for k, v in row.items() if k.startswith("plr_m") and np.isfinite(v)]
        row["window_range"] = float(max(vals) - min(vals)) if len(vals) > 1 else np.nan
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(fm.OUT / "refwindow_fixed.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
