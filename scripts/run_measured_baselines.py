"""Baselines and a loss-function diagnostic on the 15 measured records.

  pvusa        two-stage rate with the linear PVUSA performance model, for every injection (known-truth recovery,
               same injections and truth definition as the ML models)
  stl          STL trend step on the reference's normalized daily series (injection 0)
  changepoint  one-breakpoint piecewise-linear trend step on the same series (injection 0)
  khd_mean_loss κ-HYDRA-D whose central head is fitted to the insolation-weighted MEAN (squared error) instead of the
               median (pinball); five seeds, injection 0. Tests whether the median/mean mismatch explains the
               κ-HYDRA-D - RdTools gap on the DKASC arrays.
Output: results/degradation_paper/measured_baselines_runs.csv (resumable)
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
from src.degradation import baselines, ml_plr  # noqa: E402
from src.degradation.kappa_hydra_d import estimate  # noqa: E402
from src.degradation.rdtools_pipeline import gamma_for  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

OUT = ROOT / "results/degradation_paper/measured_baselines_runs.csv"
INJECTIONS = [0.0, -0.25, -0.5, -1.0, -2.0]
COLUMNS = ["system_id", "method", "injection", "plr", "t_center_years", "val_rmse", "breakpoint_years", "plr_seed_sd",
           "coverage_80"]


def systems() -> list[str]:
    ref = pd.read_csv(ROOT / "results/degradation_paper/reference_plr.csv")
    return ref[ref.status == "ok"].system_id.tolist()


def main(only: list[str] | None = None, methods: tuple = ("pvusa", "stl", "changepoint", "khd_mean_loss")) -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    keys = set(zip(done.system_id, done.method, done.injection.astype(float)))
    for sid in systems():
        if only and sid not in only and sid.split("_")[-1] not in only:
            continue
        raw = sid.split("_", 1)[1] if sid.startswith("pvdaq_") else sid
        frame, meta = load_system(raw)
        gamma = gamma_for(meta["label"])[0]
        for inj in INJECTIONS:
            todo = [m for m in methods if (sid, m, inj) not in keys and (inj == 0 or m == "pvusa")]
            if not todo:
                continue
            f = frame
            if inj:
                f = frame.assign(power_w=ml_plr.inject(ml_plr.clean(frame), inj))
                f = f[f.power_w.notna()]
            rows = []
            for m in todo:
                row = {"system_id": sid, "method": m, "injection": inj}
                if m == "pvusa":
                    r = baselines.pvusa_plr(f, meta["dc_kw"])
                    row.update(plr=r["plr"], t_center_years=r.get("t_center_years"), val_rmse=r["val_rmse"])
                elif m in ("stl", "changepoint"):
                    daily = baselines.daily_normalized(ml_plr.clean(f), meta["dc_kw"], gamma)
                    if m == "stl":
                        row.update(plr=baselines.stl_rate(daily))
                    else:
                        cp = baselines.changepoint_rate(daily)
                        row.update(plr=cp["plr"], breakpoint_years=cp["breakpoint_years"])
                else:
                    r = estimate(f, meta["dc_kw"], gamma, seeds=(0, 1, 2, 3, 4), jackknife=False, variant="mean_loss", **hp)
                    row.update(plr=r["plr"], t_center_years=r.get("t_center_years"), val_rmse=r.get("val_rmse"),
                               plr_seed_sd=r.get("plr_seed_sd"), coverage_80=r.get("coverage_80"))
                rows.append(row)
                print(sid, m, inj, round(row["plr"], 3), flush=True)
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
    sys.stdout.flush()
    os._exit(0)
