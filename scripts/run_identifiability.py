"""H01: identifiability of the κ-HYDRA-D trend when the weather itself drifts (scenario fixed before any result).

The correction g(x) has no time or age input, but its inputs (POA, cell and ambient temperature) can drift over the
years; g could then represent part of a monotone change and bias rho. Test on the semi-synthetic generator of
run_semisynthetic.py (real weather and gaps of the four sites, 1 % noise, P_r = 5 kW):

  responses    PVWatts (datasheet gamma) and thin-film-like (misspecified for the reference)
  true rates   0 and -1 %/yr
  weather      as measured ("none"), or with an imposed monotone drift ("drift"): ambient temperature +0.2 K/yr and
               POA x (1 + 0.5 %/yr * t), i.e. about +2.4 K and +6 % over twelve years, larger than observed
               climate trends, so the test is a stress test
  seeds        0 and 1 (noise), one κ-HYDRA-D seed per record as in the other semi-synthetic tests
  methods      κ-HYDRA-D (full model) and the RdTools reference (datasheet gamma)

Profile of the training loss over rho: for two drift records (thin-film-like, -1 %/yr; Alice Springs and Arbuckle
weather) rho is fixed on a grid, g is refitted, and the insolation-weighted pinball loss on the training hours is
recorded; a single, well-defined minimum near the true rate supports identifiability of rho under these conditions.
Outputs: results/degradation_paper/identifiability_runs.csv, identifiability_profile.csv
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
sys.path.insert(0, str(ROOT / "scripts"))
import run_semisynthetic as semi  # noqa: E402
from src.degradation import kappa_hydra_d as kd  # noqa: E402
from src.degradation.rdtools_pipeline import cell_temperature, sensor_plr  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

RES = ROOT / "results/degradation_paper"
OUT = RES / "identifiability_runs.csv"
PROFILE = RES / "identifiability_profile.csv"


def drifted(frame: pd.DataFrame) -> pd.DataFrame:
    t = (frame.index - frame.index.min()).total_seconds() / (365.25 * 86400)
    f = frame.copy()
    f["t_amb_c"] = f.t_amb_c + 0.2 * t
    if "t_module_c" in f:
        f["t_module_c"] = f.t_module_c + 0.2 * t
    f["poa_wm2"] = f.poa_wm2 * (1 + 0.005 * t)
    return f


def runs(hp: dict) -> None:
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["weather", "family", "rate", "seed", "drift", "method"])
    keys = set(zip(done.weather, done.family, done.rate.astype(float), done.seed, done.drift, done.method))
    only = [a for a in sys.argv[1:] if not a.startswith("--")]      # optional: one worker per weather site
    for raw in (only or semi.WEATHER):
        frame0, meta = load_system(raw)
        wid = meta["system_id"]
        for drift in ("none", "drift"):
            frame = drifted(frame0) if drift == "drift" else frame0
            for family in ("pvwatts", "thin_film"):
                for rate in (0.0, -1.0):
                    for seed in (0, 1):
                        todo = [m for m in ("rdtools", "khd_full") if (wid, family, rate, seed, drift, m) not in keys]
                        if not todo:
                            continue
                        syn = semi.synthesize(frame, meta, family, rate, seed)
                        rows = []
                        for m in todo:
                            if m == "rdtools":
                                plr = sensor_plr(syn, 5.0, semi.G_DATASHEET)["plr"]
                            else:
                                plr = kd.estimate(syn, 5.0, semi.G_DATASHEET, seeds=(seed,), jackknife=False, **hp)["plr"]
                            rows.append({"weather": wid, "family": family, "rate": rate, "seed": seed, "drift": drift,
                                         "method": m, "plr": plr, "error": plr - rate})
                            print(wid, drift, family, rate, seed, m, round(plr, 3), flush=True)
                        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def pinball(net, data, device) -> float:
    train = data[~data.val]
    pred = kd._predict(net, train, device)
    y = train.kappa.to_numpy()[:, None]
    q = np.array(kd.QUANTILES)[None, :]
    err = y - pred
    w = (train.poa / train.poa.mean()).to_numpy()[:, None]
    return float((np.maximum(q * err, (q - 1) * err) * w).mean())


def profile(hp: dict) -> None:
    if PROFILE.exists():
        return
    device = kd.DEVICE
    rows = []
    for raw in ("dkasc_7", "2107"):
        frame0, meta = load_system(raw)
        syn = semi.synthesize(drifted(frame0), meta, "thin_film", -1.0, 0)
        data = kd.design(syn, 5.0, semi.G_DATASHEET)
        t_mid = float(data.t_mid.iloc[0])
        to_first = lambda rho: rho / (1.0 + rho * (0.5 - t_mid))  # noqa: E731
        free_net, free_rho = kd._fit(data, 0, hp["hidden"], hp["lr"], hp["steps"], device)
        for delta in np.linspace(-0.01, 0.01, 9):                  # fixed rho around the free fit, +-1 %/yr
            rho = free_rho + delta
            net, _ = kd._fit(data, 0, hp["hidden"], hp["lr"], hp["steps"], device, fixed_rate=rho)
            rows.append({"weather": meta["system_id"], "rate_first_year_pct": 100 * to_first(rho),
                         "free_fit_rate_pct": 100 * to_first(free_rho), "true_rate_pct": -1.0,
                         "train_pinball": pinball(net, data, device)})
            print(meta["system_id"], round(rows[-1]["rate_first_year_pct"], 3), rows[-1]["train_pinball"], flush=True)
    pd.DataFrame(rows).to_csv(PROFILE, index=False)


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    torch.manual_seed(0)
    runs(hp)
    if "--profile" in sys.argv or not sys.argv[1:]:
        profile(hp)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
