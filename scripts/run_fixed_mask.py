"""Injection experiment with a fixed filter mask (protocol IP1-IP3, dated 2026-10-02, committed before the runs).

The filter mask (RdTools filters with the reference normalization) and the target hygiene of every system are
computed once on the unmodified record and applied to every injected rate and every method, so all estimators are
scored on identical hours. Main benchmark: 18 systems at 5 locations (the 15 long sensor-grade systems and the
three long expansion systems with on-site POA: Golden 10 and 33, Las Vegas 1278).

  python scripts/run_fixed_mask.py rd        reference (technology-typical gamma)           CPU
  python scripts/run_fixed_mask.py ml        two-stage models (MODELS=comma list), 5 seeds  GPU/CPU
  python scripts/run_fixed_mask.py mlfull    XGBoost and TCN trained on the whole record    information-equal
  python scripts/run_fixed_mask.py kappa     kappa-HYDRA-D: full (5 seeds, jackknife at r = 0, learned PI saved),
                                             no_physics and no_correction at r = 0 and -1
  python scripts/run_fixed_mask.py kappa24   kappa-HYDRA-D with g trained on the first 24 data months (rho = 0),
                                             then frozen while rho is fitted on the whole record

Resumable; RUN_TAG=<suffix>, SHARD=i/n for parallel workers. Outputs: results/fixed_mask/*.csv, pi/*.parquet
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
from src.degradation import ml_plr, perf_models as pm  # noqa: E402
from src.degradation.rdtools_pipeline import fixed_mask, gamma_for, sensor_plr  # noqa: E402

OUT = ROOT / "results/fixed_mask"
SEEDS = [0, 1, 2, 3, 4]
INJECTIONS = [0.0, -0.25, -0.5, -1.0, -2.0]
EXPANSION = ["pvdaq_10", "pvdaq_33", "pvdaq_1278"]
_CACHE: dict = {}


def systems() -> list[str]:
    ref = pd.read_csv(ROOT / "results/degradation_paper/reference_plr.csv")
    s = list(ref[ref.status == "ok"].system_id) + EXPANSION
    shard = os.environ.get("SHARD", "")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        s = [x for k, x in enumerate(s) if k % n == i]
    return s[::-1] if os.environ.get("REVERSE") == "1" else s


def load(sid: str):
    if sid not in _CACHE:
        _CACHE.clear()
        raw = sid.replace("pvdaq_", "")
        if sid in EXPANSION:
            from src.degradation import expansion as ex
            frame, meta = ex.load(raw)
        else:
            from src.degradation.tier_a import load_system
            frame, meta = load_system(raw)
        base = ml_plr.clean(frame)
        base = base[base.power_w.notna()]
        gamma, _ = gamma_for(meta["label"])
        yoy_mask = fixed_mask(base, meta["dc_kw"], gamma)
        healthy = ml_plr.healthy_hours(base, meta["dc_kw"])
        _CACHE[sid] = (base, meta, gamma, yoy_mask, healthy)
    return _CACHE[sid]


def injected(base: pd.DataFrame, inj: float) -> pd.DataFrame:
    return base.assign(power_w=ml_plr.inject(base, inj)) if inj else base


def out_file(kind: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{kind}_runs{os.environ.get('RUN_TAG', '')}.csv"


def done(kind: str, keys: list[str]) -> set:
    parts = [pd.read_csv(p) for p in OUT.glob(f"{kind}_runs*.csv")] if OUT.exists() else []
    if not parts:
        return set()
    d = pd.concat(parts).dropna(subset=["plr"])
    return set(map(tuple, d[keys].astype(str).to_numpy()))


def append(kind: str, row: dict) -> None:
    f = out_file(kind)
    cols = list(pd.read_csv(f, nrows=0).columns) if f.exists() else list(row)
    pd.DataFrame([row]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)


def run_rd() -> None:
    keys = done("rd", ["system_id", "injection"])
    for sid in systems():
        for inj in INJECTIONS:
            if (sid, str(inj)) in keys:
                continue
            base, meta, gamma, yoy_mask, _ = load(sid)
            r = sensor_plr(injected(base, inj), meta["dc_kw"], gamma, mask=yoy_mask)
            append("rd", {"system_id": sid, "injection": inj, "gamma": gamma,
                          **{k: r.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years", "t_center_years",
                                                   "points_kept_fraction")}})
            print("rd", sid, inj, round(r.get("plr", np.nan), 3), flush=True)


def run_ml(full_record: bool = False) -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    default = "xgboost,tcn" if full_record else ",".join(pm.ALL_MODELS)
    models = os.environ.get("MODELS", default).split(",")
    kind = "mlfull" if full_record else "ml"
    for sid in systems():
        for inj in INJECTIONS:
            keys = done(kind, ["system_id", "model", "seed", "injection"])
            todo = [(m, s) for m in models for s in SEEDS if (sid, m, str(s), str(inj)) not in keys]
            if not todo:
                continue
            base, meta, gamma, yoy_mask, healthy = load(sid)
            frame = injected(base, inj)
            d = ml_plr.prepare(frame, meta["dc_kw"], healthy=healthy, full_record=full_record)
            train, val = (d["x"][d["train"]], d["y"][d["train"]]), (d["x"][d["val"]], d["y"][d["val"]])
            for model, seed in todo:
                pred, info = pm.fit_predict(model, hp[model]["params"], seed, train, val, d["x"])
                sc = ml_plr.scores(pred[d["val"]], d["y"][d["val"]])
                r = ml_plr.plr_from_prediction(frame, meta["dc_kw"], pred, d["stamps"], mask=yoy_mask)
                append(kind, {"system_id": sid, "model": model, "seed": seed, "injection": inj,
                              "regime": "full record" if full_record else "24 months",
                              "n_train": int(d["train"].sum()), **{f"val_{k}": v for k, v in sc.items()},
                              **{k: r.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years",
                                                       "t_center_years", "points_kept_fraction")}})
                print(kind, sid, inj, model, seed, round(r.get("plr", np.nan), 3), flush=True)


def run_kappa(correction_months: int | None = None) -> None:
    from src.degradation.kappa_hydra_d import estimate
    from src.degradation.rdtools_pipeline import cell_temperature, yoy_from_normalized
    params = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    kind = "kappa24" if correction_months else "kappa"
    jobs = ([("full", inj) for inj in INJECTIONS] if correction_months else
            [("full", inj) for inj in INJECTIONS] + [(v, inj) for v in ("no_physics", "no_correction") for inj in (0.0, -1.0)])
    for sid in systems():
        for variant, inj in jobs:
            keys = done(kind, ["system_id", "variant", "injection"])
            if (sid, variant, str(inj)) in keys:
                continue
            base, meta, gamma, yoy_mask, healthy = load(sid)
            frame = injected(base, inj)
            want_pi = variant == "full" and inj == 0.0 and not correction_months
            r = estimate(frame, meta["dc_kw"], gamma, seeds=tuple(SEEDS), variant=variant,
                         jackknife=(variant == "full" and inj == 0.0 and not correction_months),
                         mask=yoy_mask & healthy, pi_mask=yoy_mask if want_pi else None,
                         correction_months=correction_months, **params)
            r.pop("_val", None)
            pi = r.pop("_pi", None)
            row = {"system_id": sid, "variant": variant, "injection": inj,
                   "regime": "24 months" if correction_months else "full record", **r}
            if pi is not None:                      # 2 x 2 cell C: learned normalization + YoY, same hours as A
                (OUT / "pi").mkdir(parents=True, exist_ok=True)
                pi.rename("pi").to_frame().to_parquet(OUT / "pi" / f"{sid}.parquet")
                poa = base.poa_wm2.clip(lower=0)
                tcell = cell_temperature(poa, base.get("t_amb_c"), base.get("t_module_c"))
                c = yoy_from_normalized(pi.reindex(base.index), poa, tcell, base.power_w.clip(lower=0), mask=yoy_mask)
                row["plr_knorm_yoy"] = c.get("plr")
            append(kind, row)
            print(kind, sid, variant, inj, round(r.get("plr", np.nan), 3), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"rd": run_rd, "ml": run_ml, "mlfull": lambda: run_ml(True), "kappa": run_kappa,
     "kappa24": lambda: run_kappa(24)}[mode]()
    sys.stdout.flush()
    os._exit(0)
