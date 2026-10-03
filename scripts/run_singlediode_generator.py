"""Independent known-rate generator (protocol P5, committed 2026-10-02 before the runs).

Power is generated with the pvlib CEC single-diode model, not with the multiplicative structure of any estimator:
  effective irradiance = (POA beam x IAM(physical) + POA diffuse) x spectral factor (First Solar model; precipitable
  water fixed at 1.4 cm because the cached ERA5 has no humidity; air-mass dependence retained),
  cell temperature = SAPM open rack (wind 1 m/s),
  degradation through the diode parameters: R_s grows by 5 |r| % per year and I_L falls linearly, with the I_L rate
  solved so that P_mp at standard conditions changes by r %/yr on average over the record (truth = r),
  day-to-day AR(1) multiplicative noise (phi = 0.7, sigma 1 %), a 1 % irradiance-sensor scale error (per record),
  inverter clipping at DC/AC = 1.25.
Modules: one mono-c-Si and one CdTe module of the CEC database. Weather: ERA5 2015-2022 at the 20 weather sites.
Records: 20 sites x 2 modules x 2 rates x 3 seeds = 240.

  python scripts/run_singlediode_generator.py run      METHODS=comma list (default: all), SHARD=i/n, RUN_TAG
  python scripts/run_singlediode_generator.py summary  -> results/singlediode/tables/
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
import run_semisynthetic_climate as climate  # noqa: E402
from src.degradation import expansion as ex  # noqa: E402

OUT = ROOT / "results/singlediode"
RATES, SEEDS, DC_KW = (-0.5, -1.0), (0, 1, 2), 5.0
MODULES = {"mono-Si": "Canadian_Solar_Inc__CS6K_275M", "CdTe": "First_Solar__Inc__FS_6385"}
SPECTRAL = {"mono-Si": "monosi", "CdTe": "cdte"}
ML = ("xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer")
ALL_METHODS = ("rdtools", *ML, "khd_full", "khd_24", "xgboost_full", "tcn_full")
_CEC = None


def module_params(tech: str) -> pd.Series:
    global _CEC
    if _CEC is None:
        _CEC = pvlib.pvsystem.retrieve_sam("CECMod")
    name = MODULES[tech]
    if name not in _CEC:
        cand = [c for c in _CEC.columns if _CEC[c]["Technology"] == ("Mono-c-Si" if tech == "mono-Si" else "CdTe")]
        name = cand[0]
    return _CEC[name].rename(name)


def stc_pmp(p: pd.Series, il_f: np.ndarray, rs_f: np.ndarray) -> np.ndarray:
    il, i0, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_cec(
        np.full(len(il_f), 1000.0), np.full(len(il_f), 25.0), p.alpha_sc, p.a_ref, p.I_L_ref, p.I_o_ref,
        p.R_sh_ref, p.R_s, p.Adjust)
    return pvlib.pvsystem.singlediode(il * il_f, i0, rs * rs_f, rsh, nnsvth)["p_mp"].to_numpy()


def degradation_factors(p: pd.Series, rate: float, years: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R_s grows by 5|r| % per year; the linear I_L loss is solved so that the average annual change of the STC
    P_mp over the record equals ``rate`` (%/yr)."""
    k_rs = 0.05 * abs(rate)
    target = 1 + rate / 100 * years
    lo, hi = 0.0, 0.05
    p0 = stc_pmp(p, np.array([1.0]), np.array([1.0]))[0]
    for _ in range(60):                        # bisection on the I_L rate (fraction per year)
        k_il = (lo + hi) / 2
        ratio = stc_pmp(p, np.array([1 - k_il * years]), np.array([1 + k_rs * years]))[0] / p0
        lo, hi = (k_il, hi) if ratio > target else (lo, k_il)
    k_il = (lo + hi) / 2
    return 1 - k_il * t, 1 + k_rs * t


def synthesize(row: pd.Series, tech: str, rate: float, seed: int) -> tuple[pd.DataFrame, dict]:
    meta = climate.site_meta(row)
    era = ex.era5(meta, climate.START, climate.END, ex.ERA5_VARIABLES)
    mid = era.index + pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(meta["lat"], meta["lon"])
    sun = loc.get_solarposition(mid)
    zen, az = sun.apparent_zenith.to_numpy(), sun.azimuth.to_numpy()
    am_rel = pvlib.atmosphere.get_relative_airmass(zen)
    poa = pvlib.irradiance.get_total_irradiance(
        meta["tilt"], meta["azimuth"], zen, az, era.direct_normal_irradiance.to_numpy(),
        era.shortwave_radiation.to_numpy(), era.diffuse_radiation.to_numpy(),
        dni_extra=pvlib.irradiance.get_extra_radiation(mid).to_numpy(), airmass=am_rel, albedo=0.2, model="perez")
    aoi = pvlib.irradiance.aoi(meta["tilt"], meta["azimuth"], zen, az)
    iam = np.nan_to_num(pvlib.iam.physical(aoi), nan=0.0)
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel)
    spec = pvlib.spectrum.spectral_factor_firstsolar(1.4, np.nan_to_num(am_abs, nan=10.0), SPECTRAL[tech])
    beam = np.nan_to_num(np.asarray(poa["poa_direct"], float))
    diffuse = np.nan_to_num(np.asarray(poa["poa_diffuse"], float))
    eff = np.clip((beam * iam + diffuse) * np.nan_to_num(spec, nan=1.0), 0, None)
    g_poa = np.clip(np.nan_to_num(np.asarray(poa["poa_global"], float)), 0, None)
    tamb = era.temperature_2m.to_numpy()
    tcell = pvlib.temperature.sapm_cell(g_poa, tamb, 1.0, **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"])
    p = module_params(tech)
    t = (era.index - era.index[0]).total_seconds().to_numpy() / (365.25 * 86400)
    il_f, rs_f = degradation_factors(p, rate, float(t.max()), t)
    il, i0, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_cec(np.maximum(eff, 1.0), tcell, p.alpha_sc, p.a_ref,
                                                            p.I_L_ref, p.I_o_ref, p.R_sh_ref, p.R_s, p.Adjust)
    pmp = pvlib.pvsystem.singlediode(il * il_f, i0, rs * rs_f, rsh, nnsvth)["p_mp"].to_numpy()
    n_mod = DC_KW * 1000.0 / float(p.STC)
    pdc = np.where(eff > 1.0, pmp * n_mod, 0.0)
    rng = np.random.default_rng(10_000 + seed)
    days = pd.Index(era.index.floor("D")).unique()
    e = np.zeros(len(days))
    for k in range(1, len(days)):               # day-to-day AR(1) multiplicative noise, marginal sigma 1 %
        e[k] = 0.7 * e[k - 1] + 0.01 * np.sqrt(1 - 0.7 ** 2) * rng.standard_normal()
    noise = pd.Series(e, index=days).reindex(era.index.floor("D")).to_numpy()
    power = np.minimum(pdc * (1 + noise), DC_KW * 1000.0 / 1.25)          # inverter clipping, DC/AC = 1.25
    scale = 1 + 0.01 * rng.standard_normal()                               # irradiance-sensor scale error
    index = (era.index + pd.Timedelta(hours=meta["std_offset_h"])).tz_localize(None)
    frame = pd.DataFrame({"power_w": power, "poa_wm2": g_poa * scale, "t_amb_c": tamb}, index=index)
    frame = frame[~frame.index.duplicated()].sort_index()
    pmp_stc = stc_pmp(p, np.array([1.0, il_f[-1]]), np.array([1.0, rs_f[-1]]))
    truth = float(100 * (pmp_stc[1] / pmp_stc[0] - 1) / t.max())
    return frame, {"dc_kw": DC_KW, "label": tech, "module": p.name, "truth": truth}


def runs_file() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"runs{os.environ.get('RUN_TAG', '')}.csv"


def done_keys() -> set:
    parts = [pd.read_csv(f) for f in OUT.glob("runs*.csv")] if OUT.exists() else []
    if not parts:
        return set()
    d = pd.concat(parts).dropna(subset=["plr"])
    return set(zip(d.site_id, d.module_tech, d.rate.astype(float), d.seed.astype(int), d.method))


def run(methods: tuple[str, ...]) -> None:
    from src.degradation import ml_plr, perf_models as pm
    from src.degradation.kappa_hydra_d import estimate
    from src.degradation.rdtools_pipeline import gamma_for, sensor_plr
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    table = climate.sites()
    table = table[(table.era5 == "ok")]
    rows = [r for _, r in table.iterrows()]
    shard = os.environ.get("SHARD", "")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        rows = [r for k, r in enumerate(rows) if k % n == i]
    if os.environ.get("REVERSE") == "1":
        rows = rows[::-1]
    for row in rows:
        for tech in MODULES:
            for rate in RATES:
                for seed in SEEDS:
                    keys = done_keys()
                    todo = [m for m in methods if (row.site_id, tech, float(rate), seed, m) not in keys]
                    if not todo:
                        continue
                    frame, meta = synthesize(row, tech, rate, seed)
                    gamma = gamma_for(tech)[0]
                    d, dfull = None, None
                    for m in todo:
                        res = {"ci_low": np.nan, "ci_high": np.nan}
                        if m == "rdtools":
                            r = sensor_plr(frame, DC_KW, gamma)
                            res.update(plr=r["plr"], ci_low=r["ci_low"], ci_high=r["ci_high"])
                        elif m in ML:
                            d = d or ml_plr.prepare(frame, DC_KW)
                            pred, _ = pm.fit_predict(m, hp[m]["params"], seed, (d["x"][d["train"]], d["y"][d["train"]]),
                                                     (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
                            res["plr"] = ml_plr.plr_from_prediction(frame, DC_KW, pred, d["stamps"])["plr"]
                        elif m in ("xgboost_full", "tcn_full"):
                            dfull = dfull or ml_plr.prepare(frame, DC_KW, full_record=True)
                            base = m.replace("_full", "")
                            pred, _ = pm.fit_predict(base, hp[base]["params"], seed, (dfull["x"][dfull["train"]], dfull["y"][dfull["train"]]),
                                                     (dfull["x"][dfull["val"]], dfull["y"][dfull["val"]]), dfull["x"])
                            res["plr"] = ml_plr.plr_from_prediction(frame, DC_KW, pred, dfull["stamps"])["plr"]
                        elif m == "khd_full":            # five seeds and the year jackknife: rate interval coverage
                            r = estimate(frame, DC_KW, gamma, seeds=(0, 1, 2, 3, 4), jackknife=True,
                                         **hp["kappa_hydra_d"]["params"])
                            res.update(plr=r["plr"], ci_low=r.get("ci_low"), ci_high=r.get("ci_high"))
                        elif m == "khd_24":
                            r = estimate(frame, DC_KW, gamma, seeds=(seed,), jackknife=False, correction_months=24,
                                         **hp["kappa_hydra_d"]["params"])
                            res["plr"] = r["plr"]
                        out = {"site_id": row.site_id, "climate": row.climate, "country": row.country,
                               "module_tech": tech, "module": meta["module"], "rate": rate, "seed": seed, "method": m,
                               **res, "truth": meta["truth"], "error": res["plr"] - meta["truth"]}
                        f = runs_file()
                        cols = list(pd.read_csv(f, nrows=0).columns) if f.exists() else list(out)
                        pd.DataFrame([out]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)
                        print(row.site_id, tech, rate, seed, m, round(res["plr"], 3), "truth", round(meta["truth"], 3),
                              flush=True)


def summary() -> None:
    runs = pd.concat([pd.read_csv(f) for f in OUT.glob("runs*.csv")]).dropna(subset=["plr"])
    runs = runs.drop_duplicates(["site_id", "module_tech", "rate", "seed", "method"], keep="last")
    site = runs.groupby(["method", "site_id"]).error.agg(site_mae=lambda e: e.abs().mean()).reset_index()
    t = site.groupby("method").site_mae.agg(mae="mean", worst="max").reset_index()
    t["bias"] = t.method.map(runs.groupby("method").error.mean())
    t["n_records"] = t.method.map(runs.groupby("method").size())
    best = site.loc[site.groupby("site_id").site_mae.idxmin()].method.value_counts()
    t["sites_best"] = t.method.map(best).fillna(0).astype(int)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    t.sort_values("mae").to_csv(OUT / "tables/SD1_singlediode_by_method.csv", index=False)
    k = runs[(runs.method == "khd_full") & runs.ci_low.notna()]
    cov = ((k.truth >= k.ci_low) & (k.truth <= k.ci_high))
    n, c = len(k), int(cov.sum())
    z = 1.96
    p = c / n if n else np.nan
    centre = (p + z * z / (2 * n)) / (1 + z * z / n) if n else np.nan
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else np.nan
    pd.DataFrame([{"n": n, "covered": c, "coverage": p, "wilson_low": centre - half, "wilson_high": centre + half,
                   "median_width": float((k.ci_high - k.ci_low).median())}]).to_csv(
        OUT / "tables/SD2_khd_interval_coverage.csv", index=False)
    print(t.sort_values("mae").to_string(index=False))
    print(f"coverage {c}/{n}, Wilson [{centre - half:.3f}, {centre + half:.3f}], median width {(k.ci_high - k.ci_low).median():.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if mode == "run":
        sel = os.environ.get("METHODS", "")
        run(tuple(sel.split(",")) if sel else ALL_METHODS)
    else:
        summary()
    sys.stdout.flush()
    os._exit(0)
