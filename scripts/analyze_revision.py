"""Tables of the revision (protocol P1-P7 and amendment 1) from the run files. Works on partial runs (coverage is
printed). Outputs: results/fixed_mask/tables/R*.csv and results/fixed_mask/tables/revision_numbers.json

R1  injection error per method: location-weighted MAE (5 locations), per location, worst location, n
R2  Arbuckle recovery ratio (observed / expected change of Eq. (2)) per method and injected rate
R3  2 x 2 cells A-D and effects per system; first- and second-half reference trends
R4  information-equal comparison: injection (fixed mask), 20-site exact-rate records, single-diode records
R5  reference variants on the unmodified records (typical, listed, fitted, clear-sky)
R6  single-diode generator: error by method, kappa-HYDRA-D interval coverage (Wilson)
R7  identifiability profile on measured systems
R8  measured spread of the seven two-stage rates and reference-window range
R9  mixed-effects model (random location intercept, system variance component) and delta = 0.05 equivalence
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FM = ROOT / "results/fixed_mask"
SD = ROOT / "results/singlediode"
CL = ROOT / "results/climate_expansion"
TAB = FM / "tables"
DELTA = 0.05
SEVEN = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
LOCATION = {**{f"dkasc_{i}": "Alice Springs" for i in (7, 8, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21)},
            "pvdaq_1430": "Golden", "pvdaq_10": "Golden", "pvdaq_33": "Golden", "pvdaq_1423": "Henderson",
            "pvdaq_1278": "Las Vegas", "pvdaq_2107": "Arbuckle"}
INJ = [-0.25, -0.5, -1.0, -2.0]


def read(pattern: str, base: Path = FM) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(base.glob(pattern))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def expected_change(r_inj: float, d: float, t_c: float) -> float:
    """Eq. (2): expected change of a first-year-recentred rate under an injected trend (all in %/yr, years)."""
    return r_inj * (1 + d * (2 * t_c - 1) / 100) / (1 + r_inj / 200)


def injection_table() -> pd.DataFrame:
    """One row per system x method x injected rate: rate, change, expected change, error."""
    rows = []
    rd = read("rd_runs*.csv")
    if len(rd):
        rd = rd.drop_duplicates(["system_id", "injection"], keep="last").assign(method="reference")
        rows.append(rd[["system_id", "method", "injection", "plr", "t_center_years"]])
    for name, pat in (("ml", "ml_runs*.csv"), ("mlfull", "mlfull_runs*.csv")):
        ml = read(pat)
        if len(ml):
            ml = ml.drop_duplicates(["system_id", "model", "seed", "injection"], keep="last")
            g = ml.groupby(["system_id", "model", "injection"]).agg(plr=("plr", "mean"), t_center_years=("t_center_years", "mean"),
                                                                       n_seeds=("seed", "nunique")).reset_index()
            g["method"] = g.model + ("_full" if name == "mlfull" else "")
            rows.append(g[["system_id", "method", "injection", "plr", "t_center_years", "n_seeds"]])
    for name, pat in (("kappa", "kappa_runs*.csv"), ("kappa24", "kappa24_runs*.csv")):
        k = read(pat)
        if len(k):
            k = k.drop_duplicates(["system_id", "variant", "injection"], keep="last")
            k["method"] = np.where(k.variant == "full", "khd" if name == "kappa" else "khd_24", "khd_" + k.variant)
            k = k.rename(columns={"t_center_years": "t_center_years"})
            rows.append(k[["system_id", "method", "injection", "plr", "t_center_years"]])
    rv = FM / "reference_variants_injection.csv"
    if rv.exists():
        v = pd.read_csv(rv)
        for col, tc, m in (("plr_fitted", "t_center_fitted", "reference_fitted"), ("plr_clearsky", "t_center_clearsky", "reference_clearsky")):
            if col in v:
                rows.append(v[["system_id", "injection", col, tc]].rename(columns={col: "plr", tc: "t_center_years"}).assign(method=m))
    t = pd.concat(rows, ignore_index=True).dropna(subset=["plr"])
    base = t[t.injection == 0.0].set_index(["system_id", "method"])
    out = t[t.injection != 0.0].copy()
    key = list(zip(out.system_id, out.method))
    out["d"] = [base.plr.get(k, np.nan) for k in key]
    out["t_c"] = [base.t_center_years.get(k, np.nan) for k in key]
    out["observed"] = out.plr - out.d
    out["expected"] = [expected_change(r, d, tc) for r, d, tc in zip(out.injection, out.d, out.t_c)]
    out["error"] = out.observed - out.expected
    out["recovery"] = out.observed / out.expected
    out["location"] = out.system_id.map(LOCATION)
    return out.dropna(subset=["error"])


def location_weighted(e: pd.DataFrame) -> pd.DataFrame:
    sys_mae = e.groupby(["method", "location", "system_id"]).error.agg(lambda x: x.abs().mean()).rename("mae").reset_index()
    loc = sys_mae.groupby(["method", "location"]).mae.mean().unstack()
    t = pd.DataFrame({"loc_weighted_mae": loc.mean(axis=1), "worst_location": loc.max(axis=1),
                      "system_mae": sys_mae.groupby("method").mae.mean(),
                      "n_systems": sys_mae.groupby("method").system_id.nunique(),
                      "n_rows": e.groupby("method").size()})
    return t.join(loc).sort_values("loc_weighted_mae")


def two_by_two(e_all: pd.DataFrame) -> pd.DataFrame:
    rd = read("rd_runs*.csv").drop_duplicates(["system_id", "injection"], keep="last")
    k = read("kappa_runs*.csv").drop_duplicates(["system_id", "variant", "injection"], keep="last")
    a = rd[rd.injection == 0].set_index("system_id").plr
    b = k[(k.variant == "no_correction") & (k.injection == 0)].set_index("system_id").plr
    full = k[(k.variant == "full") & (k.injection == 0)].set_index("system_id")
    t = pd.DataFrame({"A": a, "B": b, "C": full.get("plr_knorm_yoy"), "D": full.plr}).dropna()
    t["normalization_effect"] = ((t.C - t.A) + (t.D - t.B)) / 2
    t["estimator_effect"] = ((t.B - t.A) + (t.D - t.C)) / 2
    t["D_minus_A"] = t.D - t.A
    halves = FM / "half_trends.csv"
    if halves.exists():
        t = t.join(pd.read_csv(halves).set_index("system_id"))
    t["location"] = t.index.map(LOCATION)
    return t


def half_trends() -> None:
    """Reference YoY on the first and second half of each unmodified record (fixed mask)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_fixed_mask as fm
    from src.degradation.rdtools_pipeline import sensor_plr
    rows = []
    for sid in fm.systems():
        base, meta, gamma, ym, _ = fm.load(sid)
        mid = base.index.min() + (base.index.max() - base.index.min()) / 2
        r1 = sensor_plr(base[base.index <= mid], meta["dc_kw"], gamma, mask=ym)
        r2 = sensor_plr(base[base.index > mid], meta["dc_kw"], gamma, mask=ym)
        rows.append({"system_id": sid, "first_half": r1["plr"], "second_half": r2["plr"],
                     "half_difference": abs(r2["plr"] - r1["plr"])})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(FM / "half_trends.csv", index=False)


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return np.nan, np.nan
    p = c / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def single_diode() -> tuple[pd.DataFrame, dict]:
    r = read("runs*.csv", SD)
    if not len(r):
        return pd.DataFrame(), {}
    r = r.dropna(subset=["plr"]).drop_duplicates(["site_id", "module_tech", "rate", "seed", "method"], keep="last")
    site = r.groupby(["method", "site_id"]).error.agg(lambda e: e.abs().mean()).rename("site_mae").reset_index()
    t = site.groupby("method").site_mae.agg(mae="mean", worst_site="max")
    t["bias"] = r.groupby("method").error.mean()
    t["n_records"] = r.groupby("method").size()
    for tech in ("mono-Si", "CdTe"):
        t[f"mae_{tech}"] = r[r.module_tech == tech].groupby("method").error.agg(lambda e: e.abs().mean())
    k = r[(r.method == "khd_full") & r.ci_low.notna()]
    cov = {}
    for name, sub in (("khd_full", k), ("reference", r[(r.method == "rdtools") & r.ci_low.notna()])):
        c = int(((sub.truth >= sub.ci_low) & (sub.truth <= sub.ci_high)).sum())
        lo, hi = wilson(c, len(sub))
        cov[name] = {"n": int(len(sub)), "covered": c, "coverage": c / len(sub) if len(sub) else np.nan,
                     "wilson_low": lo, "wilson_high": hi, "median_width": float((sub.ci_high - sub.ci_low).median()) if len(sub) else np.nan}
    return t.sort_values("mae"), cov


def climate_info_equal() -> pd.DataFrame:
    base = read("semisynthetic_climate_runs*.csv", CL)
    ie = read("info_equal_runs*.csv", CL)
    r = pd.concat([base, ie], ignore_index=True).dropna(subset=["plr"])
    r = r.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    site = r.groupby(["method", "site_id"]).error.agg(lambda e: e.abs().mean()).rename("site_mae").reset_index()
    t = site.groupby("method").site_mae.agg(mae="mean", worst_site="max")
    t["n_records"] = r.groupby("method").size()
    return t


def mixed_model(e: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """|error| ~ method, random intercept for location, variance component for system within location."""
    import statsmodels.formula.api as smf
    n_sys = e.groupby("method").system_id.nunique()
    methods = [m for m in methods if n_sys.get(m, 0) == n_sys.max()]    # complete methods only (balanced design)
    d = e[e.method.isin(methods)].copy()
    d["abs_error"] = d.error.abs()
    d["method"] = pd.Categorical(d.method, categories=methods)
    out = []
    try:
        m = smf.mixedlm("abs_error ~ C(method)", d, groups=d["location"], re_formula="1",
                        vc_formula={"system": "0 + C(system_id)"}).fit(reml=True)
        ci = m.conf_int()
        for name in m.params.index:
            if name.startswith("C(method)"):
                lo, hi = ci.loc[name]
                out.append({"contrast": f"{name.split('T.')[-1].rstrip(']')} - {methods[0]}", "estimate": m.params[name],
                            "ci_low": lo, "ci_high": hi, "p_value": m.pvalues[name],
                            "equivalent_delta": bool(lo > -DELTA and hi < DELTA)})
    except Exception as exc:  # reported, not hidden
        out.append({"contrast": f"failed: {exc}"})
    return pd.DataFrame(out)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["halves"]:
        half_trends()
        return
    numbers = {}
    e = injection_table()
    e.to_csv(TAB / "R0_injection_rows.csv", index=False)
    r1 = location_weighted(e)
    r1.to_csv(TAB / "R1_injection_by_method.csv")
    print("R1\n", r1.round(3).to_string())
    arb = e[e.system_id == "pvdaq_2107"].pivot_table(index="method", columns="injection", values="recovery")
    arb.to_csv(TAB / "R2_arbuckle_recovery.csv")
    print("R2\n", arb.round(3).to_string())
    rec = e.pivot_table(index="system_id", columns="method", values="recovery", aggfunc=["min", "max"])
    rec.to_csv(TAB / "R2b_recovery_all_systems.csv")
    t22 = two_by_two(e)
    t22.to_csv(TAB / "R3_two_by_two.csv")
    print("R3\n", t22.round(3).to_string())
    if len(t22):
        numbers["two_by_two"] = {"n": int(len(t22)),
                                 "norm_median_abs": float(t22.normalization_effect.abs().median()),
                                 "norm_max_abs": float(t22.normalization_effect.abs().max()),
                                 "est_median_abs": float(t22.estimator_effect.abs().median()),
                                 "est_max_abs": float(t22.estimator_effect.abs().max())}
    # information-equal
    info = r1.reindex(["reference", "khd", "khd_24", "xgboost", "xgboost_full", "tcn", "tcn_full"])[["loc_weighted_mae", "worst_location", "n_systems"]]
    info = info.join(climate_info_equal().rename(index={"khd_full": "khd", "rdtools": "reference"}).add_prefix("exact_"), how="left")
    sd_t, cov = single_diode()
    if len(sd_t):
        info = info.join(sd_t.rename(index={"khd_full": "khd", "rdtools": "reference"})[["mae", "worst_site"]].add_prefix("sd_"), how="left")
        sd_t.to_csv(TAB / "R6_singlediode_by_method.csv")
        print("R6\n", sd_t.round(3).to_string(), "\n", cov)
        numbers["sd_coverage"] = cov
    info.to_csv(TAB / "R4_information_equal.csv")
    print("R4\n", info.round(3).to_string())
    refv = FM / "reference_variants.csv"
    if refv.exists():
        rv = pd.read_csv(refv)
        rv.to_csv(TAB / "R5_reference_variants.csv", index=False)
        if "plr_clearsky" in rv:
            dk = rv[rv.system_id.str.startswith("dkasc")]
            numbers["clearsky_minus_sensor_dkasc"] = [float((dk.plr_clearsky - dk.plr_typical).min()),
                                                     float((dk.plr_clearsky - dk.plr_typical).max())]
    prof = FM / "profile_summary.csv"
    if prof.exists():
        p = pd.read_csv(prof)
        p.to_csv(TAB / "R7_profile.csv", index=False)
        w = p.loc[p.width_5pct.idxmax()]
        numbers["profile"] = {"n": int(len(p)), "single_minimum": int(p.single_interior_minimum.sum()),
                              "widest_system": w.system_id, "widest_width": float(w.width_5pct),
                              "median_width": float(p.width_5pct.median())}
    # measured spread (unmodified records, mean of seeds) and reference window
    zero = pd.concat([pd.DataFrame()] + [read("ml_runs*.csv")]).drop_duplicates(["system_id", "model", "seed", "injection"], keep="last")
    if len(zero):
        z = zero[zero.injection == 0].groupby(["system_id", "model"]).plr.mean().unstack()
        z = z[[m for m in SEVEN if m in z]]
        spread = (z.max(axis=1) - z.min(axis=1)).rename("spread_seven")
        spread.to_csv(TAB / "R8_spread.csv")
        numbers["spread"] = {"n": int(spread.notna().sum()), "median": float(spread.median()), "max": float(spread.max()),
                             "models_complete": int((z.notna().sum(axis=1) == 7).sum())}
    rw = FM / "refwindow_fixed.csv"
    if rw.exists():
        w = pd.read_csv(rw)
        numbers["refwindow"] = {"n": int(w.window_range.notna().sum()), "median": float(w.window_range.median()),
                                "max": float(w.window_range.max())}
    mm = mixed_model(e, ["reference", "khd", "khd_24", "xgboost", "xgboost_full", "lstm", "gru", "tcn", "tcn_full",
                         "patchtst", "informer", "random_forest"])
    mm.to_csv(TAB / "R9_mixed_model.csv", index=False)
    print("R9\n", mm.round(4).to_string())
    numbers["R1"] = r1[["loc_weighted_mae", "worst_location", "n_systems"]].round(4).to_dict(orient="index")
    (TAB / "revision_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
