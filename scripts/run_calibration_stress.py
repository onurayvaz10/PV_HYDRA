"""Calibration of the rate intervals under other noise levels, temporal dependence and heteroscedastic noise
(protocol amendment 4, committed before the runs).

Records: the exact-rate generator of run_semisynthetic.py with the PVWatts response (the estimators' response is
correct, so only the noise differs), ERA5 weather 2015-2022 at the 20 weather sites, rate -1 %/yr, noise seeds 0
and 1. Noise modes (multiplicative, on hourly power):
  iid_0.5   independent hourly noise, sigma 0.5 %
  iid_1     independent hourly noise, sigma 1 %  (the noise of the main exact-rate records)
  iid_3     independent hourly noise, sigma 3 %
  ar1_day   day-to-day AR(1) noise (phi 0.9, marginal sigma 1 %) plus independent hourly noise 0.5 %
  hetero    independent hourly noise with sigma 1 % x sqrt(1000 / POA), POA floored at 50 W/m2 (larger at low light)
Methods: reference (RdTools default bootstrap interval), the reference with a 30-day circular block bootstrap of
its year-on-year slopes, and kappa-HYDRA-D (five seeds and the leave-one-year-out jackknife).
Truth: the first-year rate r / (1 + r / 200) (Eq. 3).

  python scripts/run_calibration_stress.py run        SHARD=i/n, RUN_TAG, METHODS=rdtools,khd_full
  python scripts/run_calibration_stress.py summary    -> results/calibration_stress/tables/K1_*.csv
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_semisynthetic_climate as climate  # noqa: E402
from run_block_bootstrap import block_interval  # noqa: E402
from src.degradation.truth_convention import first_year  # noqa: E402

OUT = ROOT / "results/calibration_stress"
MODES = ("iid_0.5", "iid_1", "iid_3", "ar1_day", "hetero")
RATE, SEEDS, DC_KW, G = -1.0, (0, 1), 5.0, -0.0040


def noise_seed(site_id: str, mode: str, seed: int) -> int:
    """Independent noise per site, mode and seed (amendment 4c: the first run seeded by mode and seed only, so every
    site received the same noise sequence; kept in results/calibration_stress_v1_shared_seed)."""
    return zlib.crc32(f"{site_id}|{mode}|{seed}".encode())


def synthesize(frame: pd.DataFrame, mode: str, rate: float, seed: int, site_id: str) -> pd.DataFrame:
    from src.degradation import ml_plr
    from src.degradation.rdtools_pipeline import cell_temperature
    f = ml_plr.clean(frame)[["poa_wm2", "t_amb_c"]].copy()
    poa = f.poa_wm2.clip(lower=0)
    tc = cell_temperature(poa, f.t_amb_c, None)
    t = (f.index - f.index.min()).total_seconds() / (365.25 * 86400)
    rng = np.random.default_rng(noise_seed(site_id, mode, seed))
    n = len(f)
    if mode.startswith("iid_"):
        eps = rng.normal(0, float(mode.split("_")[1]) / 100, n)
    elif mode == "ar1_day":
        days = pd.Index(f.index.floor("D")).unique()
        e = np.zeros(len(days))
        e[0] = 0.01 * rng.standard_normal()
        for k in range(1, len(days)):
            e[k] = 0.9 * e[k - 1] + 0.01 * np.sqrt(1 - 0.9 ** 2) * rng.standard_normal()
        eps = pd.Series(e, index=days).reindex(f.index.floor("D")).to_numpy() + rng.normal(0, 0.005, n)
    elif mode == "hetero":
        sigma = 0.01 * np.sqrt(1000.0 / poa.clip(lower=50).to_numpy())
        eps = rng.normal(0, 1, n) * sigma
    else:
        raise ValueError(mode)
    p = DC_KW * 1000 * poa / 1000 * (1 + G * (tc - 25)) * (1 + rate / 100 * t) * (1 + eps)
    f["power_w"] = p.clip(lower=0).where(poa.notna())
    return f


def runs_file() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"runs{os.environ.get('RUN_TAG', '')}.csv"


def done_keys() -> set:
    parts = [pd.read_csv(p) for p in OUT.glob("runs*.csv")] if OUT.exists() else []
    if not parts:
        return set()
    d = pd.concat(parts).dropna(subset=["plr"])
    return set(zip(d.site_id, d["mode"], d.seed.astype(int), d.method))


def run(methods: tuple[str, ...]) -> None:
    from src.degradation.kappa_hydra_d import estimate
    from src.degradation.rdtools_pipeline import sensor_plr
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    table = climate.sites()
    table = table[(table.era5 == "ok") & ~table.climate.astype(str).str.startswith("pending")]
    rows = [r for _, r in table.iterrows()]
    shard = os.environ.get("SHARD", "")
    jobs = [(row, mode, seed) for row in rows for mode in MODES for seed in SEEDS]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    if os.environ.get("REVERSE") == "1":
        jobs = jobs[::-1]
    truth = float(first_year(RATE))
    cached = {}
    for row, mode, seed in jobs:
        keys = done_keys()
        todo = [m for m in methods if (row.site_id, mode, seed, m) not in keys]
        if "rdtools" in todo:
            todo.append("rdtools_block")
        todo = [m for m in dict.fromkeys(todo) if (row.site_id, mode, seed, m) not in keys]
        if not todo:
            continue
        if row.site_id not in cached:
            cached.clear()
            cached[row.site_id] = climate.weather(row)
        frame, _ = cached[row.site_id]
        syn = synthesize(frame, mode, RATE, seed, row.site_id)
        results = {}
        if "rdtools" in todo or "rdtools_block" in todo:
            os.environ["PV_RETURN_YOY"] = "1"
            r = sensor_plr(syn, DC_KW, G)
            results["rdtools"] = (r["plr"], r["ci_low"], r["ci_high"])
            lo, hi = block_interval(r["_yoy_values"], seed)
            results["rdtools_block"] = (r["plr"], lo, hi)
        if "khd_full" in todo:
            r = estimate(syn, DC_KW, G, seeds=(0, 1, 2, 3, 4), jackknife=True, **hp)
            results["khd_full"] = (r["plr"], r.get("ci_low"), r.get("ci_high"))
        for m in todo:
            plr, lo, hi = results[m]
            out = {"site_id": row.site_id, "climate": row.climate, "country": row.country, "mode": mode, "rate": RATE,
                   "seed": seed, "method": m, "plr": plr, "ci_low": lo, "ci_high": hi, "truth": truth,
                   "error": plr - truth, "covered": bool(lo <= truth <= hi)}
            f = runs_file()
            cols = list(pd.read_csv(f, nrows=0).columns) if f.exists() else list(out)
            pd.DataFrame([out]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(row.site_id, mode, seed, m, round(plr, 3), [round(lo, 3), round(hi, 3)], out["covered"], flush=True)


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = c / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summary() -> None:
    d = pd.concat([pd.read_csv(p) for p in OUT.glob("runs*.csv")]).dropna(subset=["plr"])
    d = d.drop_duplicates(["site_id", "mode", "seed", "method"], keep="last")
    rows = []
    for (mode, m), g in d.groupby(["mode", "method"]):
        c, n = int(g.covered.sum()), len(g)
        lo, hi = wilson(c, n)
        width = g.ci_high - g.ci_low
        miss = np.maximum(g.ci_low - g.truth, 0) + np.maximum(g.truth - g.ci_high, 0)
        site = g.groupby("site_id").error.agg(lambda e: e.abs().mean())
        rows.append({"mode": mode, "method": m, "n": n, "sites": g.site_id.nunique(), "covered": c, "coverage": c / n,
                     "wilson_low": lo, "wilson_high": hi, "median_width": width.median(),
                     "interval_score": float((width + 2 / 0.05 * miss).mean()), "mae": site.mean(),
                     "bias": g.error.mean()})
    t = pd.DataFrame(rows)
    t["mode"] = pd.Categorical(t["mode"], MODES, ordered=True)
    t = t.sort_values(["mode", "method"])
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "tables/K1_interval_calibration.csv", index=False)
    print(t.to_string(index=False))
    # independence check: noise-driven error should not share one sign across all sites (amendment 4c)
    same_sign = d.groupby(["mode", "method"]).error.apply(lambda e: float(max((e > 0).mean(), (e < 0).mean())))
    print("largest share of errors with one sign:", same_sign.round(2).to_dict())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if mode == "run":
        sel = os.environ.get("METHODS", "")
        run(tuple(sel.split(",")) if sel else ("rdtools", "khd_full"))
    else:
        summary()
    sys.stdout.flush()
    os._exit(0)
