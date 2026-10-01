"""Two-stage ML PLR for the degradation paper (RQ1) — HPO, seeds, synthetic injection.

  python scripts/run_ml_plr.py hpo   equal budget: 20 Optuna-TPE trials per model, objective =
                                     mean validation RMSE (blocked weeks inside the reference
                                     period) on the three development systems, seed 0.
                                     PLR never enters the objective (no tuning on the estimand).
  python scripts/run_ml_plr.py run   every system with an RdTools reference PLR: 7 models x 5 seeds
                                     x injection r in {0, -0.25, -0.5, -1, -2} %/yr. Resumable.

Outputs: configs/perf_models_hpo.json, results/degradation_paper/ml_plr_runs.csv,
         results/degradation_paper/rdtools_injected.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import ml_plr, perf_models as pm
from src.degradation.rdtools_pipeline import gamma_for, sensor_plr
from src.degradation.tier_a import load_system

OUT = ROOT / "results/degradation_paper"
HPO_FILE = ROOT / "configs/perf_models_hpo.json"
DEV_SYSTEMS = ["dkasc_7", "dkasc_12", "dkasc_14"]      # CdTe, mono-Si, poly-Si (pre-specified)
SEEDS = [0, 1, 2, 3, 4]
INJECTIONS = [0.0, -0.25, -0.5, -1.0, -2.0]
TRIALS = 20
KD_COLUMNS = ["system_id", "climate", "injection", "status", "variant", "plr", "plr_seed_sd", "plr_mid_reference",
              "t_center_years", "rows", "span_years", "val_rmse", "val_mae", "val_r2", "coverage_80",
              "interval_width_80", "plr_se", "ci_low", "ci_high", "jackknife_years", "fit_seconds"]


def space(trial, model: str) -> dict:
    if model == "xgboost":
        return {"n_estimators": 1500, "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20)}
    if model == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 8, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", [1.0, 0.5, "sqrt"])}
    return {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024])}


_LOADED: dict = {}


def _load(system_id: str) -> tuple[pd.DataFrame, dict]:
    if system_id not in _LOADED:
        _LOADED.clear()                       # one system in memory at a time
        _LOADED[system_id] = load_system(system_id)
    frame, meta = _LOADED[system_id]
    return frame.copy(), dict(meta)


def _data(system_id: str, injection: float = 0.0) -> tuple[pd.DataFrame, dict, dict]:
    frame, meta = _load(system_id)
    if injection:
        frame = frame.assign(power_w=ml_plr.inject(ml_plr.clean(frame), injection))
        frame = frame[frame.power_w.notna()]
    return frame, meta, ml_plr.prepare(frame, meta["dc_kw"])


def _split(d: dict):
    return (d["x"][d["train"]], d["y"][d["train"]]), (d["x"][d["val"]], d["y"][d["val"]])


def hpo() -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = {s: _data(s)[2] for s in DEV_SYSTEMS}
    best = json.loads(HPO_FILE.read_text()) if HPO_FILE.exists() else {}
    for model in pm.ALL_MODELS:
        if model in best:
            continue
        started = time.perf_counter()

        def objective(trial):
            params = space(trial, model)
            rmse = []
            for d in data.values():
                train, val = _split(d)
                pred, _ = pm.fit_predict(model, params, 0, train, val, val[0])
                rmse.append(ml_plr.scores(pred, val[1])["rmse"])
            return float(np.mean(rmse))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=TRIALS)
        params = space(optuna.trial.FixedTrial(study.best_params), model)
        best[model] = {"params": params, "val_rmse": study.best_value, "trials": TRIALS,
                       "hpo_seconds": time.perf_counter() - started}
        HPO_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(model, best[model], flush=True)
    if "kappa_hydra_d" not in best:
        # Same budget and objective (validation RMSE in P/P_rated, dev systems, seed 0) for the proposed model.
        from src.degradation.kappa_hydra_d import estimate
        systems = {s: load_system(s) for s in DEV_SYSTEMS}
        started = time.perf_counter()

        def objective(trial):
            params = {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
                      "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
                      "steps": trial.suggest_categorical("steps", [2000, 3000, 5000])}
            return float(np.mean([estimate(f, m["dc_kw"], gamma_for(m["label"])[0], seeds=(0,), jackknife=False,
                                           **params)["val_rmse"] for f, m in systems.values()]))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=TRIALS)
        best["kappa_hydra_d"] = {"params": study.best_params, "val_rmse": study.best_value, "trials": TRIALS,
                                 "hpo_seconds": time.perf_counter() - started}
        HPO_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print("kappa_hydra_d", best["kappa_hydra_d"], flush=True)


def run_kappa(only: list[str] | None = None) -> None:
    """kappa-HYDRA-D: full model on every injection (5 seeds; year-jackknife CI at r = 0);
    ablations at r = 0 and r = -1 %/yr (5 seeds)."""
    from src.degradation.kappa_hydra_d import VARIANTS, estimate
    params = json.loads(HPO_FILE.read_text())["kappa_hydra_d"]["params"]
    ref = pd.read_csv(OUT / "reference_plr.csv")
    systems = [s for s in ref[ref.status == "ok"].system_id if not only or s in only]
    out = OUT / "kappa_hydra_d_runs.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["system_id", "variant", "injection"])
    keys = set(zip(done.system_id, done.variant, done.injection.astype(float)))
    for sid in systems:
        raw_id = sid.replace("pvdaq_", "")
        jobs = [("full", inj) for inj in INJECTIONS] + [(v, inj) for v in VARIANTS if v != "full" for inj in (0.0, -1.0)]
        for variant, inj in jobs:
            if (sid, variant, inj) in keys:
                continue
            frame, meta = _load(raw_id)
            if inj:
                frame = frame.assign(power_w=ml_plr.inject(ml_plr.clean(frame), inj))
                frame = frame[frame.power_w.notna()]
            gamma, _ = gamma_for(meta["label"])
            r = estimate(frame, meta["dc_kw"], gamma, seeds=tuple(SEEDS), variant=variant,
                         jackknife=(variant == "full" and inj == 0.0), **params)
            val = r.pop("_val", None)
            if val is not None and inj == 0.0:
                vp = OUT / "val_preds"
                vp.mkdir(exist_ok=True)
                np.savez_compressed(vp / f"{sid}_khd-{variant}_all.npz", **val)
            row = {"system_id": sid, "climate": meta["climate"], "injection": inj, **r}
            # Rows without jackknife lack the CI fields: align to one fixed column order, or appended
            # values shift under the header (fixed 2026-09-29; earlier file repaired).
            cols = KD_COLUMNS if not out.exists() else list(pd.read_csv(out, nrows=0).columns)
            pd.DataFrame([row]).reindex(columns=cols).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(sid, variant, inj, round(r.get("plr", np.nan), 3), round(r.get("val_rmse", np.nan), 4), flush=True)


def run_rdtools_injected(only: list[str] | None = None) -> None:
    """RdTools reference on the injected records (CPU only; can run beside GPU work)."""
    ref = pd.read_csv(OUT / "reference_plr.csv")
    systems = [s for s in ref[ref.status == "ok"].system_id if not only or s in only]
    rd_file = OUT / "rdtools_injected.csv"
    rd_done = pd.read_csv(rd_file) if rd_file.exists() else pd.DataFrame(columns=["system_id", "injection"])
    rd_keys = set(zip(rd_done.system_id, rd_done.injection.astype(float)))
    for sid in systems:
        for inj in INJECTIONS:
            if (sid, inj) in rd_keys:
                continue
            frame, meta = _load(sid.replace("pvdaq_", ""))
            if inj:
                frame = frame.assign(power_w=ml_plr.inject(ml_plr.clean(frame), inj))
                frame = frame[frame.power_w.notna()]
            gamma, _ = gamma_for(meta["label"])
            r = sensor_plr(ml_plr.clean(frame), meta["dc_kw"], gamma, soiling=False, freq_minutes=60)
            pd.DataFrame([{"system_id": sid, "injection": inj, **{k: r.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years", "t_center_years")}}]
                         ).to_csv(rd_file, mode="a", header=not rd_file.exists(), index=False)
            print(sid, inj, round(r.get("plr", np.nan), 3), flush=True)


def run(only: list[str] | None = None) -> None:
    hp = json.loads(HPO_FILE.read_text())
    ref = pd.read_csv(OUT / "reference_plr.csv")
    systems = [s for s in ref[ref.status == "ok"].system_id if not only or s in only]
    # Parallel workers (disjoint system lists) each append to their own file; completed units are
    # read from all of them, so any worker can resume any unit.
    tag = os.environ.get("RUN_TAG", "")
    runs_file, rd_file = OUT / f"ml_plr_runs{tag}.csv", OUT / "rdtools_injected.csv"
    parts = [pd.read_csv(p) for p in OUT.glob("ml_plr_runs*.csv")]
    done = pd.concat(parts) if parts else pd.DataFrame(columns=["system_id", "model", "seed", "injection"])
    done_keys = set(zip(done.system_id, done.model, done.seed, done.injection.astype(float)))
    rd_done = pd.read_csv(rd_file) if rd_file.exists() else pd.DataFrame(columns=["system_id", "injection"])
    rd_keys = set(zip(rd_done.system_id, rd_done.injection.astype(float)))
    for sid in systems:
        raw_id = sid.replace("pvdaq_", "")
        for inj in INJECTIONS:
            frame, meta, d = _data(raw_id, inj)
            soiling = str(meta["climate"]).startswith("B")
            if (sid, inj) not in rd_keys:
                gamma, _ = gamma_for(meta["label"])
                r = sensor_plr(ml_plr.clean(frame), meta["dc_kw"], gamma, soiling=False, freq_minutes=60)
                pd.DataFrame([{"system_id": sid, "injection": inj, **{k: r.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years", "t_center_years")}}]
                             ).to_csv(rd_file, mode="a", header=not rd_file.exists(), index=False)
            train, val = _split(d)
            for model in pm.ALL_MODELS:
                for seed in SEEDS:
                    if (sid, model, seed, inj) in done_keys:
                        continue
                    pred, info = pm.fit_predict(model, hp[model]["params"], seed, train, val, d["x"])
                    sc = ml_plr.scores(pred[d["val"]], d["y"][d["val"]])
                    if inj == 0.0:        # held-out predictions kept for Diebold-Mariano tests
                        vp = OUT / "val_preds"
                        vp.mkdir(exist_ok=True)
                        np.savez_compressed(vp / f"{sid}_{model}_{seed}.npz", stamps=d["stamps"][d["val"]].astype("int64"),
                                            y=d["y"][d["val"]], pred=pred[d["val"]])
                    r = ml_plr.plr_from_prediction(frame, meta["dc_kw"], pred, d["stamps"])
                    row = {"system_id": sid, "climate": meta["climate"], "model": model, "seed": seed, "injection": inj,
                           "ref_end": str(d["ref_end"]), "n_train": int(d["train"].sum()), "n_val": int(d["val"].sum()),
                           **{f"val_{k}": v for k, v in sc.items()},
                           **{k: r.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years", "t_center_years", "points_kept_fraction")},
                           **info}
                    pd.DataFrame([row]).to_csv(runs_file, mode="a", header=not runs_file.exists(), index=False)
                    print(sid, inj, model, seed, round(sc["rmse"], 4), round(r.get("plr", np.nan), 3), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "hpo":
        hpo()
    elif mode == "rd":
        run_rdtools_injected(sys.argv[2:] or None)
    elif mode == "kappa":
        run_kappa(sys.argv[2:] or None)
    else:
        run(sys.argv[2:] or None)
    sys.stdout.flush()
    os._exit(0)
