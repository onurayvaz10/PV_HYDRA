"""Revision checks (H04, H05, H07, H08) computed from the existing result files; no model is refitted.

T27_site_dependence.csv    known-truth recovery MAE per method with different weightings: all 15 arrays, the four
                           sites weighted equally, DKASC only, US sites only, each site left out, and the three
                           development systems excluded; plus the method ranking in each case.
T28_interval_quality.csv   95 % rate intervals on the semi-synthetic records: coverage, mean width and the interval
                           (Winkler) score, which penalises width and misses together (lower is better).
T29_fit_weighting.csv      retrospective (validation-week) and later-year hourly RMSE with the same equal-location
                           weights, and their ratio (Table VI).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/degradation_paper"
TAB = RES / "tables"
DEV = ["dkasc_7", "dkasc_12", "dkasc_14"]            # run_ml_plr.DEV_SYSTEMS (tuning systems)
METHODS = ["rdtools", "xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer", "khd_full"]


def site_dependence() -> pd.DataFrame:
    rec = pd.read_csv(TAB / "T2_recovery_by_system.csv")
    rec = rec[rec.method.isin(METHODS)]
    loc = pd.read_csv(TAB / "T0_sites.csv").set_index("system_id").location
    rec["location"] = rec.system_id.map(loc)
    rec["abs"] = rec.error.abs()
    per_sys = rec.groupby(["method", "location", "system_id"]).abs.mean().reset_index()
    cases = {"all 15 arrays (equal array)": per_sys,
             "four sites weighted equally": per_sys.groupby(["method", "location"]).abs.mean().reset_index(),
             "DKASC only (12 arrays)": per_sys[per_sys.location == "Alice Springs, NT"],
             "US sites only (3 systems)": per_sys[per_sys.location != "Alice Springs, NT"],
             "development systems excluded (12)": per_sys[~per_sys.system_id.isin(DEV)]}
    for site in sorted(per_sys.location.unique()):
        cases[f"without {site}"] = per_sys[per_sys.location != site]
    rows = []
    for case, d in cases.items():
        m = d.groupby("method").abs.mean()
        rank = m.rank()
        for meth in METHODS:
            rows.append({"case": case, "method": meth, "recovery_mae": m.get(meth, np.nan), "rank_by_mae": rank.get(meth, np.nan),
                         "n_units": int(d[d.method == meth].shape[0])})
    return pd.DataFrame(rows)


def interval_quality() -> pd.DataFrame:
    alpha = 0.05
    k = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "semisynthetic_ci_runs*.csv"))], ignore_index=True)
    k = k.drop_duplicates(["weather", "family", "rate", "noise_seed"], keep="last").assign(method="khd_full")
    s = pd.read_csv(RES / "semisynthetic_runs.csv")
    s = s[s.method.isin(["rdtools", "xgboost"])].rename(columns={"seed": "noise_seed"})
    d = pd.concat([k[["method", "family", "rate", "plr", "ci_low", "ci_high"]],
                   s[["method", "family", "rate", "plr", "ci_low", "ci_high"]]], ignore_index=True).dropna()
    x, lo, hi = d.rate, d.ci_low, d.ci_high
    d["width"] = hi - lo
    d["covered"] = (x >= lo) & (x <= hi)
    d["interval_score"] = d.width + 2 / alpha * ((lo - x).clip(lower=0) + (x - hi).clip(lower=0))
    return d.groupby(["method", "family"]).agg(n=("width", "size"), coverage=("covered", "mean"),
                                                mean_width=("width", "mean"), median_width=("width", "median"),
                                                interval_score=("interval_score", "mean")).reset_index()


def fit_weighting() -> pd.DataFrame:
    t = pd.read_csv(TAB / "T3b_common_domain_fit.csv")
    t["model"] = t.model.replace({"khd-full": "kappa_hydra_d"})
    loc = pd.read_csv(TAB / "T0_sites.csv").set_index("system_id").location
    t["location"] = t.system_id.map(loc)
    f = pd.read_csv(TAB / "T22_forward_by_model.csv").set_index("model")
    out = pd.DataFrame({"retro_equal_array": t.groupby("model").rmse.mean(),
                        "retro_equal_location": t.groupby(["model", "location"]).rmse.mean().groupby("model").mean(),
                        "forward_equal_location": f.rmse}).dropna()
    out["ratio_equal_location"] = out.forward_equal_location / out.retro_equal_location
    return out.reset_index().rename(columns={"index": "model"})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    a = site_dependence()
    a.to_csv(TAB / "T27_site_dependence.csv", index=False)
    print(a.pivot(index="method", columns="case", values="recovery_mae").round(3).to_string())
    b = interval_quality()
    b.to_csv(TAB / "T28_interval_quality.csv", index=False)
    print(b.round(3).to_string(index=False))
    c = fit_weighting()
    c.to_csv(TAB / "T29_fit_weighting.csv", index=False)
    print(c.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
