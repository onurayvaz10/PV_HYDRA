"""Fair reference and a response outside the PVWatts family on the 20 weather sites (protocol Y20 v2, 2026-10-02).

Part A  existing PVWatts / thin-film-like exact-rate records of run_semisynthetic_climate.py; RdTools YoY with the
        true coefficient of the generator (oracle) and with a coefficient fitted on the first 12 months.
Part B  ADR response (Driesse et al. 2021, pvlib example parameters, Faiman cell temperature), linear rate r in
        {-0.5, -1} %/yr, seeds 0-2, 1 % noise; RdTools (datasheet and fitted coefficient), PVUSA, XGBoost, TCN,
        kappa-HYDRA-D and its two ablations (one seed each).

  python scripts/run_climate_fair_reference.py a            Part A (CPU)
  python scripts/run_climate_fair_reference.py b            Part B, methods from METHODS (comma list) or all
  python scripts/run_climate_fair_reference.py summary      -> results/climate_expansion/tables/E12_fair_reference.csv

Resumable; RUN_TAG=<suffix> and SHARD=i/n for parallel workers.
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
from run_semisynthetic import G_DATASHEET, synthesize  # noqa: E402
from run_semisynthetic_gamma import fitted_gamma  # noqa: E402

OUT = ROOT / "results/climate_expansion"
RATES, SEEDS, DC_KW = (-0.5, -1.0), (0, 1, 2), 5.0
ADR = {"k_a": 1.0, "k_d": -6.0, "tc_d": 0.02, "k_rs": 0.05, "k_rsh": 0.10}
B_METHODS = ("rdtools", "rdtools_fitted_gamma", "pvusa", "xgboost", "tcn", "khd_full", "khd_no_physics",
             "khd_no_correction")
COLUMNS = ["part", "site_id", "climate", "country", "family", "rate", "seed", "method", "gamma", "plr", "truth", "error"]


def runs_file() -> Path:
    return OUT / f"fair_reference_runs{os.environ.get('RUN_TAG', '')}.csv"


def done_keys() -> set:
    parts = [pd.read_csv(p) for p in OUT.glob("fair_reference_runs*.csv")]
    if not parts:
        return set()
    d = pd.concat(parts).dropna(subset=["site_id", "method", "plr"])
    return set(zip(d.part, d.site_id, d.family, d.rate.astype(float), d.seed.astype(int), d.method))


def write(row: dict) -> None:
    f = runs_file()
    pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(f, mode="a", header=not f.exists(), index=False)
    print(row["part"], row["site_id"], row["family"], row["rate"], row["seed"], row["method"], round(row["plr"], 3),
          flush=True)


def site_rows() -> list[pd.Series]:
    t = climate.sites()
    t = t[(t.era5 == "ok") & ~t.climate.astype(str).str.startswith("pending")]
    shard = os.environ.get("SHARD", "")
    rows = [r for _, r in t.iterrows()]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        rows = [r for k, r in enumerate(rows) if k % n == i]
    return rows[::-1] if os.environ.get("REVERSE", "") == "1" else rows


def synthesize_adr(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    from src.degradation import ml_plr
    f = ml_plr.clean(frame)[["poa_wm2", "t_amb_c"]].copy()
    poa = f.poa_wm2.clip(lower=0)
    tc = pvlib.temperature.faiman(poa, f.t_amb_c, wind_speed=1.0)
    eta = pvlib.pvarray.pvefficiency_adr(poa.clip(lower=1), tc, **ADR) / pvlib.pvarray.pvefficiency_adr(1000, 25, **ADR)
    resp = pd.Series(np.asarray(eta), index=f.index).where(poa > 1, 0.0)
    t = (f.index - f.index.min()).total_seconds().to_numpy() / (365.25 * 86400)
    rng = np.random.default_rng(seed)
    p = 1000 * DC_KW * poa / 1000 * resp * (1 + rate / 100 * t) * (1 + rng.normal(0, 0.01, len(f)))
    f["power_w"] = p.clip(lower=0).where(poa.notna())
    return f


def part_a() -> None:
    from src.degradation.rdtools_pipeline import sensor_plr
    keys = done_keys()
    for row in site_rows():
        frame, meta = climate.weather(row)
        for family in ("pvwatts", "thin_film"):
            true_g = G_DATASHEET if family == "pvwatts" else 0.5 * G_DATASHEET
            for rate in RATES:
                for seed in SEEDS:
                    todo = [m for m in ("rdtools_true_gamma", "rdtools_fitted_gamma")
                            if ("A", row.site_id, family, float(rate), seed, m) not in keys]
                    if not todo:
                        continue
                    syn = synthesize(frame, meta, family, rate, seed)
                    for m in todo:
                        g = true_g if m == "rdtools_true_gamma" else fitted_gamma(syn, DC_KW)
                        plr = sensor_plr(syn, DC_KW, g)["plr"]
                        write({"part": "A", "site_id": row.site_id, "climate": row.climate, "country": row.country,
                               "family": family, "rate": rate, "seed": seed, "method": m, "gamma": g, "plr": plr,
                               "truth": rate, "error": plr - rate})


def part_b(methods: tuple[str, ...]) -> None:
    from src.degradation import baselines, ml_plr, perf_models as pm
    from src.degradation.kappa_hydra_d import estimate
    from src.degradation.rdtools_pipeline import sensor_plr
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    keys = done_keys()
    for row in site_rows():
        frame, meta = climate.weather(row)
        for rate in RATES:
            for seed in SEEDS:
                keys = done_keys()                       # other workers may have finished this record meanwhile
                todo = [m for m in methods if ("B", row.site_id, "adr", float(rate), seed, m) not in keys]
                if not todo:
                    continue
                syn = synthesize_adr(frame, rate, seed)
                d = None
                for m in todo:
                    g = G_DATASHEET
                    if m == "rdtools":
                        plr = sensor_plr(syn, DC_KW, g)["plr"]
                    elif m == "rdtools_fitted_gamma":
                        g = fitted_gamma(syn, DC_KW)
                        plr = sensor_plr(syn, DC_KW, g)["plr"]
                    elif m == "pvusa":
                        d = d or ml_plr.prepare(syn, DC_KW)
                        plr = baselines.pvusa_plr(syn, DC_KW, d)["plr"]
                    elif m in ("xgboost", "tcn"):
                        d = d or ml_plr.prepare(syn, DC_KW)
                        pred, _ = pm.fit_predict(m, hp[m]["params"], seed, (d["x"][d["train"]], d["y"][d["train"]]),
                                                 (d["x"][d["val"]], d["y"][d["val"]]), d["x"])
                        plr = ml_plr.plr_from_prediction(syn, DC_KW, pred, d["stamps"])["plr"]
                    else:
                        variant = m.replace("khd_", "") if m != "khd_full" else "full"
                        plr = estimate(syn, DC_KW, G_DATASHEET, seeds=(seed,), jackknife=False, variant=variant,
                                       **hp["kappa_hydra_d"]["params"])["plr"]
                    write({"part": "B", "site_id": row.site_id, "climate": row.climate, "country": row.country,
                           "family": "adr", "rate": rate, "seed": seed, "method": m, "gamma": g, "plr": plr,
                           "truth": rate, "error": plr - rate})


def summary() -> None:
    runs = pd.concat([pd.read_csv(p) for p in OUT.glob("fair_reference_runs*.csv")])
    runs = runs.drop_duplicates(["part", "site_id", "family", "rate", "seed", "method"], keep="last")
    site = runs.groupby(["part", "family", "method", "site_id"]).error.agg(
        site_mae=lambda e: e.abs().mean(), site_bias="mean").reset_index()
    rows = []
    for (part, fam, m), g in site.groupby(["part", "family", "method"]):
        rows.append({"part": part, "family": fam, "method": m, "n_sites": len(g), "mae": g.site_mae.mean(),
                     "bias": g.site_bias.mean(), "worst_site": g.site_mae.max(),
                     "worst_site_id": g.loc[g.site_mae.idxmax(), "site_id"],
                     "n_records": int(runs[(runs.part == part) & (runs.family == fam) & (runs.method == m)].shape[0])})
    both = site[site.part == "A"].groupby(["method", "site_id"]).site_mae.mean().reset_index()
    for m, g in both.groupby("method"):
        rows.append({"part": "A", "family": "both", "method": m, "n_sites": len(g), "mae": g.site_mae.mean(),
                     "bias": np.nan, "worst_site": g.site_mae.max(), "worst_site_id": g.loc[g.site_mae.idxmax(), "site_id"],
                     "n_records": int(runs[(runs.part == "A") & (runs.method == m)].shape[0])})
    t = pd.DataFrame(rows)
    (OUT / "tables").mkdir(exist_ok=True)
    t.to_csv(OUT / "tables/E12_fair_reference.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if mode == "a":
        part_a()
    elif mode == "b":
        sel = os.environ.get("METHODS", "")
        part_b(tuple(sel.split(",")) if sel else B_METHODS)
    elif mode == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
    sys.stdout.flush()
    os._exit(0)
