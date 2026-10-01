"""RQ4 — cross-site transfer for young plants: can PLR be estimated without 24 local months?

Leave-one-LOCATION-out (all arrays at the target's site are excluded from pre-training, so DKASC
targets are pre-trained on US systems only: a cross-climate transfer). Conditions per target:
  scratch_24   standard two-stage model, 24 local data months (RQ1 setting)
  scratch_6    same, only the first 6 local data months
  zero_shot    pre-trained on the other locations' reference periods, no local data
  finetune_6   pre-trained, then adapted with the first 6 local data months
Each condition runs on the measured record (agreement with RdTools) and with r = -1 %/yr injected
(recovery). Models: the two best two-stage models by HPO validation RMSE (argument), 5 seeds.
Output: results/degradation_paper/transfer_runs.csv (resumable).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import ml_plr, perf_models as pm
from src.degradation.tier_a import load_system

OUT = ROOT / "results/degradation_paper"
HP = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
SEEDS = [0, 1, 2, 3, 4]


def location(meta: dict) -> str:
    return "dkasc" if meta["source"] == "DKASC" else f"{round(meta['lat'], 1)}_{round(meta['lon'], 1)}"


def _prepared(sid: str, months: int, injection: float = 0.0):
    frame, meta = load_system(sid)
    if injection:
        frame = frame.assign(power_w=ml_plr.inject(ml_plr.clean(frame), injection))
        frame = frame[frame.power_w.notna()]
    return frame, meta, ml_plr.prepare(frame, meta["dc_kw"], reference_months=months)


def pretrain_pool(targets: list[str]) -> dict:
    pool = {}
    for sid in targets:
        _, meta, d = _prepared(sid.replace("pvdaq_", ""), 24)
        pool[sid] = {"loc": location(meta), "train": (d["x"][d["train"]], d["y"][d["train"]]),
                     "val": (d["x"][d["val"]], d["y"][d["val"]])}
    return pool


_PRE_CACHE: dict = {}


def fit_pretrained(model: str, seed: int, pool: dict, exclude_loc: str):
    """The pre-training pool depends only on the target's location (never on the target's injection),
    so one fit per (model, seed, location) serves every target at that location and both injections."""
    key = (model, seed, exclude_loc)
    if key not in _PRE_CACHE:
        _PRE_CACHE[key] = _fit_pretrained(model, seed, pool, exclude_loc)
    return _PRE_CACHE[key]


def _fit_pretrained(model: str, seed: int, pool: dict, exclude_loc: str):
    parts = [v for v in pool.values() if v["loc"] != exclude_loc]
    train = tuple(np.concatenate([p["train"][i] for p in parts]) for i in (0, 1))
    val = tuple(np.concatenate([p["val"][i] for p in parts]) for i in (0, 1))
    return pm.fit_model(model, HP[model]["params"], seed, train, val)


def main(models: list[str]) -> None:
    ref = pd.read_csv(OUT / "reference_plr.csv")
    systems = list(ref[ref.status == "ok"].system_id)
    out = OUT / "transfer_runs.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["system_id", "model", "seed", "condition", "injection"])
    keys = set(zip(done.system_id, done.model, done.seed, done.condition, done.injection.astype(float)))
    pool = pretrain_pool(systems)
    for sid in systems:
        raw = sid.replace("pvdaq_", "")
        for inj in (0.0, -1.0):
            frame, meta, d24 = _prepared(raw, 24, inj)
            d6 = ml_plr.prepare(frame, meta["dc_kw"], reference_months=6)
            loc = location(meta)
            for model in models:
                for seed in SEEDS:
                    todo = [c for c in ("scratch_24", "scratch_6", "zero_shot", "finetune_6")
                            if (sid, model, seed, c, inj) not in keys]
                    if not todo:
                        continue
                    pre = fit_pretrained(model, seed, pool, loc) if {"zero_shot", "finetune_6"} & set(todo) else None
                    for cond in todo:
                        d = d6 if cond in ("scratch_6", "finetune_6") else d24
                        train, val = (d["x"][d["train"]], d["y"][d["train"]]), (d["x"][d["val"]], d["y"][d["val"]])
                        if cond.startswith("scratch"):
                            est = pm.fit_model(model, HP[model]["params"], seed, train, val)
                        elif cond == "zero_shot":
                            est = pre
                        else:
                            est = pm.finetune(model, pre, HP[model]["params"], seed, train, val)
                        # d6 and d24 share samples (same windows); only the masks differ. All conditions are
                        # scored on the 24-month held-out weeks, which no condition trains on.
                        pred = pm.predict(model, est, d24["x"])
                        sc = ml_plr.scores(pred[d24["val"]], d24["y"][d24["val"]])
                        r = ml_plr.plr_from_prediction(frame, meta["dc_kw"], pred, d24["stamps"])
                        row = {"system_id": sid, "location": loc, "climate": meta["climate"], "model": model, "seed": seed,
                               "condition": cond, "injection": inj, "n_local_train": int(d["train"].sum()) if cond != "zero_shot" else 0,
                               "val_rmse": sc["rmse"], "plr": r.get("plr"), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                               "t_center_years": r.get("t_center_years")}
                        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                        print(sid, inj, model, seed, cond, round(sc["rmse"], 4), round(r.get("plr", np.nan), 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["xgboost", "tcn"])
    sys.stdout.flush()
    os._exit(0)
