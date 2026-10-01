"""Compute-cost table (guideline): parameters, training and inference time per model, measured in
isolation (nothing else running) on one development system with the tuned hyper-parameters,
5 seeds. Hardware is recorded. Output: results/degradation_paper/tables/T7_compute_cost.csv
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import ml_plr, perf_models as pm
from src.degradation.kappa_hydra_d import estimate
from src.degradation.rdtools_pipeline import gamma_for, sensor_plr
from src.degradation.tier_a import load_system

SYSTEM = "dkasc_7"


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    frame, meta = load_system(SYSTEM)
    d = ml_plr.prepare(frame, meta["dc_kw"])
    train, val = (d["x"][d["train"]], d["y"][d["train"]]), (d["x"][d["val"]], d["y"][d["val"]])
    rows = []
    for model in pm.ALL_MODELS:
        for seed in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _, info = pm.fit_predict(model, hp[model]["params"], seed, train, val, d["x"])
            rows.append({"model": model, "seed": seed, **info})
    gamma, _ = gamma_for(meta["label"])
    for seed in range(5):
        t0 = time.perf_counter()
        r = estimate(frame, meta["dc_kw"], gamma, seeds=(seed,), jackknife=False, **hp["kappa_hydra_d"]["params"])
        h = int(hp["kappa_hydra_d"]["params"]["hidden"])
        rows.append({"model": "kappa_hydra_d", "seed": seed, "train_seconds": time.perf_counter() - t0,
                     # two hidden layers (7->h->h), 3 quantile outputs, plus the scalar rate r
                     "parameters": (7 * h + h) + (h * h + h) + (h * 3 + 3) + 1})
    t0 = time.perf_counter()
    sensor_plr(ml_plr.clean(frame), meta["dc_kw"], gamma)
    rows.append({"model": "rdtools_yoy", "seed": 0, "train_seconds": time.perf_counter() - t0, "parameters": 0})
    df = pd.DataFrame(rows)
    table = df.groupby("model").agg(parameters=("parameters", "median"), train_s_mean=("train_seconds", "mean"),
                                    train_s_sd=("train_seconds", "std"),
                                    inference_ms_per_1k=("inference_seconds",
                                                         lambda s: np.nan if s.isna().all() else 1000 * s.mean())).reset_index()
    if "inference_samples" in df:
        n = df.groupby("model").inference_samples.first()
        table["inference_ms_per_1k"] = table.apply(lambda r: r.inference_ms_per_1k / n.get(r.model, np.nan) * 1000
                                                   if np.isfinite(r.inference_ms_per_1k) else np.nan, axis=1)
    table["hpo_seconds"] = table.model.map(lambda m: hp.get(m, {}).get("hpo_seconds"))
    table["hardware"] = f"{platform.processor()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    table["n_train"] = len(train[1])
    out = ROOT / "results/degradation_paper/tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "T7_compute_cost.csv", index=False)
    print(table.to_string())


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
