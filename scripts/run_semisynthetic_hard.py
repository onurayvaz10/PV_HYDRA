"""Harder semi-synthetic tests: the true system violates the assumptions of κ-HYDRA-D (and of the other methods).

Real POA and temperature of the four weather sites of run_semisynthetic.py, 3 noise seeds, 1 % noise, P_r = 5 kW.
Scenarios (trend multiplier m(t), t in years from the start; T = record length):
  knee         accelerating loss: -0.5 %/yr until T/2, then -1.5 %/yr (non-linear trajectory)
  step         -0.5 %/yr plus a 3 % step loss at T/2 (e.g. a failed string or a replaced inverter)
  soiling      -1 %/yr times an unobserved soiling cycle: 0.2 %/day accumulation, capped at 10 %, reset by random
               cleaning events (probability 1/30 per day, seeded); not an input of any model
  non_pvwatts  -1 %/yr with a response outside the PVWatts family: ADR efficiency model (Driesse et al., IEEE JPV
               2021; pvlib example parameters k_a=1, k_d=-6, tc_d=0.02, k_rs=0.05, k_rsh=0.10) at Faiman cell
               temperature
The generating response is PVWatts (datasheet gamma) except in non_pvwatts.
Truth: r_avg = 100 (m_trend(T) - 1) / T (%/yr, average annual change relative to the start; soiling excluded, it
is a stationary confounder). For knee and step no single linear rate describes the record; r_avg is the summary a
linear-rate estimator should approach, and the error against it shows how each method summarizes the trajectory.
Methods: RdTools YoY (datasheet gamma), two-stage XGBoost, TCN and PVUSA, κ-HYDRA-D (one seed), STL and
change-point trend steps on the reference's normalized daily series.
Output: results/degradation_paper/semisynthetic_hard_runs{RUN_TAG}.csv (resumable)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_semisynthetic import G_DATASHEET, SEEDS, WEATHER  # noqa: E402
from src.degradation import baselines, ml_plr, perf_models as pm  # noqa: E402
from src.degradation.kappa_hydra_d import estimate  # noqa: E402
from src.degradation.rdtools_pipeline import cell_temperature, sensor_plr  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

TAG = os.environ.get("RUN_TAG", "")
OUT = ROOT / f"results/degradation_paper/semisynthetic_hard_runs{TAG}.csv"
SCENARIOS = ["knee", "step", "soiling", "non_pvwatts"]
METHODS = ["rdtools", "xgboost", "tcn", "pvusa", "khd_full", "stl", "changepoint"]
ADR = {"k_a": 1.0, "k_d": -6.0, "tc_d": 0.02, "k_rs": 0.05, "k_rsh": 0.10}
COLUMNS = ["weather", "climate", "scenario", "seed", "method", "plr", "truth_avg", "error", "breakpoint_years"]


def multiplier(scenario: str, index: pd.DatetimeIndex, seed: int) -> tuple[np.ndarray, np.ndarray]:
    t = (index - index.min()).total_seconds().to_numpy() / (365.25 * 86400)
    T = t.max()
    if scenario == "knee":
        trend = np.where(t < T / 2, 1 - 0.005 * t, 1 - 0.005 * T / 2 - 0.015 * (t - T / 2))
    elif scenario == "step":
        trend = (1 - 0.005 * t) * np.where(t >= T / 2, 0.97, 1.0)
    else:
        trend = 1 - 0.01 * t
    soil = np.ones_like(t)
    if scenario == "soiling":
        days = pd.date_range(index.min().floor("D"), index.max().floor("D"), freq="D")
        rng = np.random.default_rng(1000 + seed)
        level, s = 1.0, []
        for _ in days:
            level = 1.0 if rng.random() < 1 / 30 else max(0.9, level - 0.002)
            s.append(level)
        soil = pd.Series(s, index=days).reindex(index.floor("D")).to_numpy()
    return trend, soil


def synthesize(frame: pd.DataFrame, meta: dict, scenario: str, seed: int) -> tuple[pd.DataFrame, float]:
    f = ml_plr.clean(frame)[["poa_wm2", "t_amb_c"] + (["t_module_c"] if "t_module_c" in frame else [])].copy()
    poa = f.poa_wm2.clip(lower=0)
    if scenario == "non_pvwatts":
        tc = pvlib.temperature.faiman(poa, f.t_amb_c, wind_speed=1.0)
        eta = pvlib.pvarray.pvefficiency_adr(poa.clip(lower=1), tc, **ADR) / pvlib.pvarray.pvefficiency_adr(1000, 25, **ADR)
        resp = pd.Series(np.asarray(eta), index=f.index).where(poa > 1, 0.0)
    else:
        tc = cell_temperature(poa, f.get("t_amb_c"), f.get("t_module_c"))
        resp = 1 + G_DATASHEET * (tc - 25)
    trend, soil = multiplier(scenario, f.index, seed)
    rng = np.random.default_rng(seed)
    p = 5000.0 * poa / 1000 * resp * trend * soil * (1 + rng.normal(0, 0.01, len(f)))
    f["power_w"] = p.clip(lower=0).where(poa.notna())
    years = (f.index.max() - f.index.min()).total_seconds() / (365.25 * 86400)
    return f, float(100 * (trend[-1] - 1) / years)


def run_methods(syn: pd.DataFrame, seed: int, hp: dict, todo: list[str]) -> list[dict]:
    rows = []
    d = None
    for method in todo:
        bp = np.nan
        if method == "rdtools":
            plr = sensor_plr(syn, 5.0, G_DATASHEET)["plr"]
        elif method in ("xgboost", "tcn"):
            d = d or ml_plr.prepare(syn, 5.0)
            pred, _ = pm.fit_predict(method, hp[method]["params"], seed, (d["x"][d["train"]], d["y"][d["train"]]),
                                     (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
            plr = ml_plr.plr_from_prediction(syn, 5.0, pred, d["stamps"])["plr"]
        elif method == "pvusa":
            d = d or ml_plr.prepare(syn, 5.0)
            plr = baselines.pvusa_plr(syn, 5.0, d)["plr"]
        elif method == "khd_full":
            plr = estimate(syn, 5.0, G_DATASHEET, seeds=(seed,), jackknife=False, **hp["kappa_hydra_d"]["params"])["plr"]
        else:
            daily = baselines.daily_normalized(syn, 5.0, G_DATASHEET)
            if method == "stl":
                plr = baselines.stl_rate(daily)
            else:
                cp = baselines.changepoint_rate(daily)
                plr, bp = cp["plr"], cp["breakpoint_years"]
        rows.append({"method": method, "plr": plr, "breakpoint_years": bp})
    return rows


def main(sites: list[str] | None = None, scenarios: list[str] | None = None, methods: list[str] | None = None) -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    keys = set(zip(done.weather, done.scenario, done.seed, done.method))
    for raw in (sites or WEATHER):
        frame, meta = load_system(raw)
        wid = meta["system_id"]
        for scenario in (scenarios or SCENARIOS):
            for seed in SEEDS:
                todo = [m for m in (methods or METHODS) if (wid, scenario, seed, m) not in keys]
                if not todo:
                    continue
                syn, truth = synthesize(frame, meta, scenario, seed)
                rows = run_methods(syn, seed, hp, todo)
                for r in rows:
                    r.update({"weather": wid, "climate": meta["climate"], "scenario": scenario, "seed": seed,
                              "truth_avg": truth, "error": r["plr"] - truth})
                    print(wid, scenario, seed, r["method"], round(r["plr"], 3), "truth", round(truth, 3), flush=True)
                pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    sites = [a for a in args if not a.startswith("--")] or None
    main(sites)
    sys.stdout.flush()
    os._exit(0)
