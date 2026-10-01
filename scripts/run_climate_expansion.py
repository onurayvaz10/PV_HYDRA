"""Climate expansion, step 3: the rate pipeline on the screened Tier-S and Tier-R systems.

Same protocol as scripts/run_ml_plr.py (HPO parameters of configs/perf_models_hpo.json, seeds 0-4, injections
0, -0.25, -0.5, -1, -2 %/yr); every row carries its tier (S or R) and role (main or sensitivity, rule 4), so Tier R
is never pooled with Tier S without a tier term. Only systems whose screen is final ("pass" in
results/climate_expansion/selection.csv with status final) are run: no rate on an incomplete screen.

  python scripts/run_climate_expansion.py queue           job list -> results/climate_expansion/queues/
  python scripts/run_climate_expansion.py rd [ids]        RdTools reference + injections (CPU)
  python scripts/run_climate_expansion.py kappa [ids]     kappa-HYDRA-D, 5 seeds (CPU or GPU)
  python scripts/run_climate_expansion.py ml [ids] [--models=xgboost,random_forest]
                                                          ML performance models (default all 7; GPU for the networks)

Run scope (plan, 2026-09-30): main-role systems get every step; sensitivity systems ranked 4-6 in their location
get rd and kappa only; lower-ranked sensitivity systems are not run.

Resumable; RUN_TAG=<suffix> lets parallel workers (disjoint system ids) write separate files.
Outputs: results/climate_expansion/rates_rdtools.csv, rates_kappa.csv, rates_ml*.csv
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
from src.degradation import expansion as ex  # noqa: E402

OUT = ROOT / "results/climate_expansion"
# Full protocol: seeds 0-4 and injections 0, -0.25, -0.5, -1, -2 %/yr. REDUCED=1 (CPU-only fallback, declared as a
# scope reduction in the plan): seed 0 and injections 0, -1 %/yr; a later full run fills in the rest (resumable).
REDUCED = os.environ.get("REDUCED", "") == "1"
SEEDS = [0] if REDUCED else [0, 1, 2, 3, 4]
INJECTIONS = [0.0, -1.0] if REDUCED else [0.0, -0.25, -0.5, -1.0, -2.0]


def selected(only: list[str] | None = None, provisional: bool = False, step: str = "rd") -> pd.DataFrame:
    sel = pd.read_csv(OUT / "selection.csv")
    if not provisional:
        sel = sel[sel.status == "final"]
    in_scope = (sel.role == "main") | ((step != "ml") & (sel.rank_in_location <= 6))
    sel = sel[in_scope]
    if only:
        sel = sel[sel.system_id.isin(only) | sel.raw_id.astype(str).isin(only)]
    return sel


_LOADED: dict = {}


def data(system_id: str, injection: float = 0.0):
    from src.degradation import ml_plr
    if system_id not in _LOADED:
        _LOADED.clear()
        sel = pd.read_csv(OUT / "selection.csv").set_index("system_id")
        raw = sel.raw_id.get(system_id) if "raw_id" in sel else None
        _LOADED[system_id] = ex.load(str(raw) if isinstance(raw, str) else system_id.replace("pvdaq_", ""))
    frame, meta = _LOADED[system_id]
    frame, meta = frame.copy(), dict(meta)
    if injection:
        frame = frame.assign(power_w=ml_plr.inject(ml_plr.clean(frame), injection))
        frame = frame[frame.power_w.notna()]
    return frame, meta


def _append(path: Path, row: dict) -> None:
    cols = list(pd.read_csv(path, nrows=0).columns) if path.exists() else list(row)
    pd.DataFrame([row]).reindex(columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def _done(pattern: str, keys: list[str]) -> set:
    parts = [pd.read_csv(p) for p in OUT.glob(pattern)]
    if not parts:
        return set()
    done = pd.concat(parts).dropna(subset=keys)            # a crash can leave a partial row
    done["injection"] = done.injection.astype(float)
    return set(map(tuple, done[keys].to_numpy().tolist()))


def run_rd(sel: pd.DataFrame) -> None:
    from src.degradation import ml_plr
    from src.degradation.rdtools_pipeline import gamma_for, sensor_plr
    path = OUT / f"rates_rdtools{os.environ.get('RUN_TAG', '')}.csv"
    done = _done("rates_rdtools*.csv", ["system_id", "injection"])
    for r in sel.itertuples():
        for inj in INJECTIONS:
            if (r.system_id, inj) in done:
                continue
            frame, meta = data(r.system_id, inj)
            gamma, _ = gamma_for(meta["label"])
            res = sensor_plr(ml_plr.clean(frame), meta["dc_kw"], gamma, soiling=False, freq_minutes=60)
            _append(path, {"system_id": r.system_id, "tier": r.tier, "role": r.role, "location": r.location,
                           "climate": meta["climate"], "irradiance_basis": meta["irradiance_basis"],
                           "temperature_basis": meta["temperature_basis"], "injection": inj,
                           **{k: res.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years", "t_center_years")}})
            print(r.system_id, inj, round(res.get("plr", np.nan), 3), flush=True)


def run_kappa(sel: pd.DataFrame) -> None:
    from src.degradation.kappa_hydra_d import estimate
    from src.degradation.rdtools_pipeline import gamma_for
    params = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    path = OUT / f"rates_kappa{os.environ.get('RUN_TAG', '')}.csv"
    done = _done("rates_kappa*.csv", ["system_id", "injection"])
    for r in sel.itertuples():
        for inj in INJECTIONS:
            if (r.system_id, inj) in done:
                continue
            frame, meta = data(r.system_id, inj)
            res = estimate(frame, meta["dc_kw"], gamma_for(meta["label"])[0], seeds=tuple(SEEDS),
                           jackknife=(inj == 0.0), **params)
            res.pop("_val", None)
            _append(path, {"system_id": r.system_id, "tier": r.tier, "role": r.role, "location": r.location,
                           "climate": meta["climate"], "injection": inj, "variant": "full", **res})
            print(r.system_id, inj, round(res.get("plr", np.nan), 3), flush=True)


def run_ml(sel: pd.DataFrame, models: tuple[str, ...] | None = None) -> None:
    from src.degradation import ml_plr, perf_models as pm
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    path = OUT / f"rates_ml{os.environ.get('RUN_TAG', '')}.csv"
    done = _done("rates_ml*.csv", ["system_id", "model", "seed", "injection"])
    for r in sel.itertuples():
        for inj in INJECTIONS:
            todo = [(m, s) for m in (models or pm.ALL_MODELS) for s in SEEDS if (r.system_id, m, s, inj) not in done]
            if not todo:
                continue
            frame, meta = data(r.system_id, inj)
            short = r.system_id.startswith(("hk_", "ut_", "fmi_"))     # amendment 2026-10-01: 2-3 y records, 12-month reference
            d = ml_plr.prepare(frame, meta["dc_kw"], reference_months=12 if short else ml_plr.REFERENCE_MONTHS)
            train, val = (d["x"][d["train"]], d["y"][d["train"]]), (d["x"][d["val"]], d["y"][d["val"]])
            for model, seed in todo:
                pred, info = pm.fit_predict(model, hp[model]["params"], seed, train, val, d["x"])
                sc = ml_plr.scores(pred[d["val"]], d["y"][d["val"]])
                res = ml_plr.plr_from_prediction(frame, meta["dc_kw"], pred, d["stamps"])
                _append(path, {"system_id": r.system_id, "tier": r.tier, "role": r.role, "location": r.location,
                               "climate": meta["climate"], "model": model, "seed": seed, "injection": inj,
                               "ref_end": str(d["ref_end"]), "n_train": int(d["train"].sum()),
                               "n_val": int(d["val"].sum()), **{f"val_{k}": v for k, v in sc.items()},
                               **{k: res.get(k) for k in ("plr", "ci_low", "ci_high", "days", "span_years",
                                                          "t_center_years", "points_kept_fraction")}, **info})
                print(r.system_id, inj, model, seed, round(sc["rmse"], 4), round(res.get("plr", np.nan), 3), flush=True)


def queue() -> None:
    from src.degradation import perf_models as pm
    rows = []
    for r in selected(provisional=True, step="rd").itertuples():
        base = {"system_id": r.system_id, "tier": r.tier, "role": r.role, "location": r.location,
                "climate": r.climate, "screen_status": r.status}
        rows += [{**base, "step": "rd", "model": "rdtools", "seed": None, "injection": i, "device": "cpu"}
                 for i in INJECTIONS]
        rows += [{**base, "step": "kappa", "model": "kappa_hydra_d", "seed": "0-4", "injection": i, "device": "gpu"}
                 for i in INJECTIONS]
        if r.role == "main":
            rows += [{**base, "step": "ml", "model": m, "seed": s, "injection": i,
                      "device": "cpu" if m in pm.TREES else "gpu"}
                     for m in pm.ALL_MODELS for s in SEEDS for i in INJECTIONS]
    (OUT / "queues").mkdir(parents=True, exist_ok=True)
    jobs = pd.DataFrame(rows)
    jobs.to_csv(OUT / "queues/expansion_rate_jobs.csv", index=False)
    if jobs.empty:
        print("no candidate passed or is pending")
        return
    print(jobs.groupby(["screen_status", "tier", "step"]).size().to_string(), f"\n{len(jobs)} jobs")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "queue"
    models = next((a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")), None)
    ids = [a for a in sys.argv[2:] if not a.startswith("--")] or None
    if mode == "queue":
        queue()
    else:
        sel = selected(ids, step=mode)
        if sel.empty:
            raise SystemExit("no system with a final 'pass' screen yet (see results/climate_expansion/selection.csv)")
        if mode == "ml":
            run_ml(sel, tuple(models) if models else None)
        else:
            {"rd": run_rd, "kappa": run_kappa}[mode](sel)
    sys.stdout.flush()
    os._exit(0)
