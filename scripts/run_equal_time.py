"""Equal-TIME tuning budget, as a complement to the equal-TRIAL budget of the main benchmark.

The per-trial cost of each model is its measured 20-trial tuning time / 20 (configs/perf_models_hpo.json, same
objective and development systems). The common wall-clock budget is the median 20-trial time of the seven
performance models; each model gets n = floor(budget / per-trial cost) trials (at least 1) of the same TPE search
(seed 42). Cheap models thus get more trials, expensive ones fewer (with fewer than 10 trials TPE is still in its
random start-up phase, which is what an equal-time budget implies).

  python scripts/run_equal_time.py tune    -> configs/perf_models_hpo_equal_time.json, results/.../hpo_equal_time_trials.csv
  python scripts/run_equal_time.py rates   -> results/.../equal_time_rates.csv: for every model whose selected
                                              configuration changed, the two-stage rate on all 15 systems at
                                              injections 0 and -1 %/yr (seed 0), next to the equal-trial configuration
  python scripts/run_equal_time.py rates --seeds=1,2,3,4
                                           -> results/.../equal_time_rates_seeds.csv: the same for further seeds, so the
                                              equal-time comparison rests on five seeds like the main benchmark
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_ml_plr as rml  # noqa: E402
from src.degradation import ml_plr, perf_models as pm  # noqa: E402

HPO = ROOT / "configs/perf_models_hpo.json"
EQ = ROOT / "configs/perf_models_hpo_equal_time.json"
TRIALS_OUT = ROOT / "results/degradation_paper/hpo_equal_time_trials.csv"
RATES_OUT = ROOT / "results/degradation_paper/equal_time_rates.csv"
SEEDS_OUT = ROOT / "results/degradation_paper/equal_time_rates_seeds.csv"


def budget() -> tuple[float, dict]:
    hp = json.loads(HPO.read_text())
    per_trial = {m: hp[m]["hpo_seconds"] / hp[m]["trials"] for m in pm.ALL_MODELS}
    total = float(np.median([hp[m]["hpo_seconds"] for m in pm.ALL_MODELS]))
    return total, {m: max(1, math.floor(total / c)) for m, c in per_trial.items()}


def tune() -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    total, n_trials = budget()
    data = {s: rml._data(s)[2] for s in rml.DEV_SYSTEMS}
    out = json.loads(EQ.read_text()) if EQ.exists() else {"budget_seconds": total}
    logs = []
    for model in pm.ALL_MODELS:
        if model in out:
            continue

        def objective(trial):
            params = rml.space(trial, model)
            rmse = []
            for d in data.values():
                train, val = rml._split(d)
                pred, _ = pm.fit_predict(model, params, 0, train, val, val[0])
                rmse.append(ml_plr.scores(pred, val[1])["rmse"])
            return float(np.mean(rmse))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials[model])
        out[model] = {"params": rml.space(optuna.trial.FixedTrial(study.best_params), model),
                      "val_rmse": study.best_value, "trials": n_trials[model]}
        logs += [{"model": model, "trial": t.number, "val_rmse": t.value} for t in study.trials]
        EQ.write_text(json.dumps(out, indent=2), encoding="utf-8")
        pd.DataFrame(logs).to_csv(TRIALS_OUT, index=False)
        print(model, n_trials[model], "trials, best", round(study.best_value, 5), flush=True)


def rates(reverse: bool = False, start: int = 0, seeds: tuple[int, ...] = (0,), models: list[str] | None = None) -> None:
    hp = json.loads(HPO.read_text())
    eq = json.loads(EQ.read_text())
    changed = [m for m in pm.ALL_MODELS if eq[m]["params"] != hp[m]["params"] and (models is None or m in models)]
    print("configuration changed:", changed, flush=True)
    ref = pd.read_csv(ROOT / "results/degradation_paper/reference_plr.csv")
    order = list(ref[ref.status == "ok"].system_id)
    order = order[start:] + order[:start]                 # a third worker can start in the middle
    for sid in (order[::-1] if reverse else order):       # a second worker can run from the other end
        raw = sid.split("_", 1)[1] if sid.startswith("pvdaq_") else sid
        out = RATES_OUT if seeds == (0,) else SEEDS_OUT
        done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["system_id", "model", "injection", "seed"])
        if "seed" not in done:
            done["seed"] = 0
        keys = set(zip(done.system_id, done.model, done.injection.astype(float), done.seed.astype(int)))
        for inj in (0.0, -1.0):
            todo = [(m, sd) for sd in seeds for m in changed if (sid, m, inj, sd) not in keys]
            if not todo:
                continue
            frame, meta, d = rml._data(raw, inj)
            for m, sd in todo:
                pred, _ = pm.fit_predict(m, eq[m]["params"], sd, (d["x"][d["train"]], d["y"][d["train"]]),
                                         (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
                r = ml_plr.plr_from_prediction(frame, meta["dc_kw"], pred, d["stamps"])
                row = {"system_id": sid, "model": m, "injection": inj, "plr": r["plr"],
                       "t_center_years": r.get("t_center_years"),
                       "val_rmse": ml_plr.scores(pred[d["val"]], d["y"][d["val"]])["rmse"]}
                if out is SEEDS_OUT:
                    row["seed"] = sd
                print(sid, m, inj, sd, round(r["plr"], 3), flush=True)
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune()
    else:
        start = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--start=")), 0)
        seeds = next((tuple(int(x) for x in a.split("=")[1].split(",")) for a in sys.argv if a.startswith("--seeds=")), (0,))
        models = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--models=")), None)
        rates(reverse="--reverse" in sys.argv, start=start, seeds=seeds, models=models)
    sys.stdout.flush()
    os._exit(0)
