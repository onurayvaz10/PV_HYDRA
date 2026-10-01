"""Semi-synthetic known-truth test across climate classes (climate expansion, plan tier "Semi-synthetic").

Weather: ERA5 hourly 2015-01-01 - 2022-12-31 (Open-Meteo archive) at the sites of
configs/semisynthetic_climate_sites.csv; POA = ERA5 GHI/DNI/DHI transposed to latitude tilt facing the equator
(pvlib Perez, albedo 0.2); 2-m temperature. Climate class = Köppen-Geiger 1991-2020 raster class at the site
(never the expected label); the run requires >= 10 distinct raster classes.
Power: the generator of scripts/run_semisynthetic.py (families pvwatts and thin_film, rates -0.5 and -1.0 %/yr,
seeds 0-2, 1 % noise); truth = the injected rate (no other trend in the generator).

  python scripts/run_semisynthetic_climate.py weather          ERA5 + raster classes (CPU, network)
  python scripts/run_semisynthetic_climate.py cpu              RdTools and kappa-HYDRA-D (CPU; kappa is slow there)
  python scripts/run_semisynthetic_climate.py gpu [methods]    any of rdtools, khd_full and the 7 ML models (default:
                                                               the 7 ML models); kappa-HYDRA-D and the networks on GPU
  python scripts/run_semisynthetic_climate.py queue            job list -> results/climate_expansion/queues/
  python scripts/run_semisynthetic_climate.py summary          per-class table

Resumable; RUN_TAG=<suffix> lets parallel workers write separate run files.
Outputs: results/climate_expansion/semisynthetic_sites.csv, semisynthetic_climate_runs*.csv,
         semisynthetic_climate_by_class.csv
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
SITES = ROOT / "configs/semisynthetic_climate_sites.csv"
START, END = "2015-01-01", "2022-12-31"
FAMILIES = ("pvwatts", "thin_film")
REDUCED = os.environ.get("REDUCED", "") == "1"      # CPU-only fallback (declared in the plan): rate -1 %/yr, seed 0
RATES = (-1.0,) if REDUCED else (-0.5, -1.0)
SEEDS = (0,) if REDUCED else (0, 1, 2)
MIN_CLASSES = 10
DC_KW = 5.0
CPU_METHODS = ("rdtools", "khd_full")


def site_meta(row: pd.Series) -> dict:
    lat, lon = float(row.latitude), float(row.longitude)
    return {"system_id": f"era5_{row.site_id}", "lat": lat, "lon": lon, "tilt": round(abs(lat), 1),
            "azimuth": 180.0 if lat >= 0 else 0.0, "std_offset_h": float(round(lon / 15.0)), "dc_kw": DC_KW}


def weather(row: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Hourly frame on local solar-standard time (UTC + round(lon/15) h): poa_wm2, t_amb_c."""
    meta = site_meta(row)
    era = ex.era5(meta, START, END, ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
                                     "temperature_2m"))
    poa = ex.transpose(era, meta)
    index = (era.index + pd.Timedelta(hours=meta["std_offset_h"])).tz_localize(None)
    frame = pd.DataFrame({"poa_wm2": poa.to_numpy(), "t_amb_c": era.temperature_2m.to_numpy()}, index=index)
    return frame[~frame.index.duplicated()].sort_index(), meta


def sites() -> pd.DataFrame:
    table = OUT / "semisynthetic_sites.csv"
    if not table.exists():
        raise SystemExit("run the 'weather' step first (needs archive-api.open-meteo.com and the raster)")
    return pd.read_csv(table)


def step_weather() -> None:
    rows = []
    for _, row in pd.read_csv(SITES).iterrows():
        meta = site_meta(row)
        out = {"site_id": row.site_id, "name": row["name"], "latitude": row.latitude, "longitude": row.longitude,
               "expected_class": row.expected_class, "climate": ex.climate_class(meta["lat"], meta["lon"]),
               "tilt": meta["tilt"], "azimuth": meta["azimuth"], "country": row.country}
        try:
            frame, _ = weather(row)
            out.update({"hours": int(frame.poa_wm2.notna().sum()),
                        "poa_kwh_m2_yr": round(float(frame.poa_wm2.sum() / 1000 / (len(frame) / 8766)), 1),
                        "t_mean_c": round(float(frame.t_amb_c.mean()), 2), "era5": "ok"})
        except Exception as exc:
            out["era5"] = f"unavailable: {type(exc).__name__}"
        rows.append(out)
        print(out, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "semisynthetic_sites.csv", index=False)
    classes = table.climate[~table.climate.astype(str).str.startswith("pending")].nunique()
    print(f"distinct raster classes: {classes} (required >= {MIN_CLASSES})")
    if classes < MIN_CLASSES:
        print("WARNING: fewer classes than required; the set is not changed after any rate has been computed")


def scenarios(table: pd.DataFrame):
    shard = os.environ.get("SHARD", "")        # "i/n": this worker takes the sites with position % n == i (parallel workers)
    for pos, (_, row) in enumerate(table.iterrows()):
        if shard and pos % int(shard.split("/")[1]) != int(shard.split("/")[0]):
            continue
        for family in FAMILIES:
            for rate in RATES:
                for seed in SEEDS:
                    yield row, family, rate, seed


def runs_file() -> Path:
    return OUT / f"semisynthetic_climate_runs{os.environ.get('RUN_TAG', '')}.csv"


def done_keys() -> set:
    parts = [pd.read_csv(p) for p in OUT.glob("semisynthetic_climate_runs*.csv")]
    if not parts:
        return set()
    done = pd.concat(parts).dropna(subset=["site_id", "family", "rate", "seed", "method"])   # a crash can leave a partial row
    return set(zip(done.site_id, done.family, done.rate.astype(float), done.seed.astype(int), done.method))


def run(methods: tuple[str, ...]) -> None:
    from scripts.run_semisynthetic import G_DATASHEET, synthesize
    from src.degradation import ml_plr, perf_models as pm
    from src.degradation.kappa_hydra_d import estimate
    from src.degradation.rdtools_pipeline import sensor_plr
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    table = sites()
    table = table[(table.era5 == "ok") & ~table.climate.astype(str).str.startswith("pending")]
    keys = done_keys()
    cached = {}
    for row, family, rate, seed in scenarios(table):
        todo = [m for m in methods if (row.site_id, family, float(rate), int(seed), m) not in keys]
        if not todo:
            continue
        if row.site_id not in cached:
            cached.clear()
            cached[row.site_id] = weather(row)
        frame, meta = cached[row.site_id]
        syn = synthesize(frame, meta, family, rate, seed)
        d = None
        for method in todo:
            if method == "rdtools":
                r = sensor_plr(syn, DC_KW, G_DATASHEET)
                res = {"plr": r["plr"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
            elif method == "khd_full":
                r = estimate(syn, DC_KW, G_DATASHEET, seeds=(seed,), jackknife=False, **hp["kappa_hydra_d"]["params"])
                res = {"plr": r["plr"], "ci_low": np.nan, "ci_high": np.nan, "coverage_80": r.get("coverage_80")}
            else:
                d = d or ml_plr.prepare(syn, DC_KW)
                pred, _ = pm.fit_predict(method, hp[method]["params"], seed, (d["x"][d["train"]], d["y"][d["train"]]),
                                         (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
                r = ml_plr.plr_from_prediction(syn, DC_KW, pred, d["stamps"])
                res = {"plr": r["plr"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
            out = {"site_id": row.site_id, "climate": row.climate, "family": family, "rate": rate, "seed": seed,
                   "method": method, **res, "error": res["plr"] - rate}
            cols = list(pd.read_csv(runs_file(), nrows=0).columns) if runs_file().exists() else list(out)
            pd.DataFrame([out]).reindex(columns=cols).to_csv(runs_file(), mode="a", header=not runs_file().exists(),
                                                             index=False)
            print(row.site_id, row.climate, family, rate, seed, method, round(res["plr"], 3), flush=True)


def queue() -> None:
    from src.degradation import perf_models as pm
    if (OUT / "semisynthetic_sites.csv").exists():
        table = sites()
        table = table[(table.era5 == "ok") & ~table.climate.astype(str).str.startswith("pending")]
        status = "ready"
    else:
        table = pd.read_csv(SITES).assign(climate=lambda t: "expected " + t.expected_class)
        status = "provisional: weather step not run (ERA5/raster unreachable in the cloud machine)"
    keys = done_keys()
    rows = [{"site_id": row.site_id, "climate": row.climate, "family": family, "rate": rate, "seed": seed,
             "method": method, "device": "cpu" if method in ("rdtools", *pm.TREES) else "gpu",
             "done": (row.site_id, family, float(rate), int(seed), method) in keys, "status": status}
            for row, family, rate, seed in scenarios(table) for method in (*CPU_METHODS, *pm.ALL_MODELS)]
    (OUT / "queues").mkdir(parents=True, exist_ok=True)
    jobs = pd.DataFrame(rows)
    jobs.to_csv(OUT / "queues/semisynthetic_climate_jobs.csv", index=False)
    print(jobs.groupby(["device", "method"]).size().to_string(), f"\n{len(jobs)} jobs, status: {status}")


def summary() -> None:
    parts = [pd.read_csv(p) for p in OUT.glob("semisynthetic_climate_runs*.csv")]
    if not parts:
        raise SystemExit("no runs yet")
    runs = pd.concat(parts).drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    by = runs.groupby(["climate", "family", "method"]).error
    table = pd.DataFrame({"bias": by.mean(), "mae": by.apply(lambda e: e.abs().mean()),
                          "max_abs": by.apply(lambda e: e.abs().max()), "n": by.size()}).reset_index()
    table.to_csv(OUT / "semisynthetic_climate_by_class.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "queue"
    if mode == "weather":
        step_weather()
    elif mode == "cpu":
        run(CPU_METHODS)
    elif mode == "gpu":
        from src.degradation import perf_models as pm
        run(tuple(sys.argv[2:]) or tuple(pm.ALL_MODELS))
    elif mode == "queue":
        queue()
    elif mode == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
    sys.stdout.flush()
    os._exit(0)
