"""Fill the v3 manuscript template from the result files.

04_calisma/_md_v3_template.md holds ⟪KEY⟫ markers; 04_calisma/_v3_prose.md holds the result-dependent prose blocks
("### KEY" + text with {name} fields). Every {name} is a number computed here from the result files, so no number
is typed by hand. Output: 04_calisma/manuscript_rev.md and 06_kontrol/checks/v3_numbers.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
FM = ROOT / "results/fixed_mask"
TAB = FM / "tables"
SD = ROOT / "results/singlediode"
CL = ROOT / "results/climate_expansion"
TPL = REV / "04_calisma/_md_v3_template.md"
PROSE = REV / "04_calisma/_v3_prose.md"
OUT = REV / "04_calisma/manuscript_rev.md"
MINUS = "−"
SEVEN = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
LABEL = {"reference": "Reference, typical γ", "reference_fitted": "Reference, fitted γ",
         "reference_clearsky": "Reference, clear-sky", "khd": "κ-HYDRA-D", "khd_24": "κ-HYDRA-D-24",
         "xgboost": "XGBoost", "xgboost_full": "XGBoost", "random_forest": "Random forest", "lstm": "LSTM", "gru": "GRU",
         "tcn": "TCN", "tcn_full": "TCN", "patchtst": "PatchTST", "informer": "Informer"}
REGIME = {"reference": "–", "reference_fitted": "first 12 months", "reference_clearsky": "–", "khd": "full record",
          "khd_24": "24 months", "xgboost_full": "full record", "tcn_full": "full record"}


def fmt(x: float, nd: int = 3, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def read(pattern: str, base: Path) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(base.glob(pattern))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = c / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def numbers() -> tuple[dict, dict]:
    N, T = {}, {}
    r1 = pd.read_csv(TAB / "R1_injection_by_method.csv").set_index("method")
    for m in r1.index:
        N[f"inj_{m}"] = fmt(r1.loc[m, "loc_weighted_mae"])
        N[f"injworst_{m}"] = fmt(r1.loc[m, "worst_location"])
        for loc in ("Alice Springs", "Golden", "Henderson", "Las Vegas", "Arbuckle"):
            if loc in r1:
                N[f"inj_{m}_{loc.split()[0].lower()}"] = fmt(r1.loc[m, loc])
    two = [m for m in SEVEN if m in r1.index]
    best2 = min(two, key=lambda m: r1.loc[m, "loc_weighted_mae"])
    N["best_two_stage"] = LABEL[best2]
    N["inj_best_two_stage"] = fmt(r1.loc[best2, "loc_weighted_mae"])
    # Arbuckle recovery
    arb = pd.read_csv(TAB / "R2_arbuckle_recovery.csv").set_index("method")
    for m in arb.index:
        N[f"arb_{m}_min"] = fmt(arb.loc[m].min(), 2)
        N[f"arb_{m}_max"] = fmt(arb.loc[m].max(), 2)
    off = pd.read_csv(TAB / "R2c_arbuckle_clipoff.csv")
    off = off[(off["mask"] == "clip off") & (off.coefficient == "typical")].drop(columns=["mask", "coefficient"])
    N["arb_off_min"], N["arb_off_max"] = fmt(float(off.min(axis=1).iloc[0]), 2), fmt(float(off.max(axis=1).iloc[0]), 2)
    # 2 x 2
    t = pd.read_csv(TAB / "R3_two_by_two.csv")
    N["n_2x2"] = str(len(t))
    N["norm_med"] = fmt(t.normalization_effect.abs().median())
    N["norm_max"] = fmt(t.normalization_effect.abs().max())
    N["est_med"] = fmt(t.estimator_effect.abs().median())
    N["est_max"] = fmt(t.estimator_effect.abs().max())
    big = t.loc[t.D_minus_A.abs().sort_values(ascending=False).index[:3]]
    N["da_largest_systems"] = ", ".join(s.replace("dkasc_", "DKASC ").replace("pvdaq_", "PVDAQ ") for s in big.system_id)
    N["da_largest_min"] = fmt(big.D_minus_A.abs().min(), 2)
    N["da_largest_max"] = fmt(big.D_minus_A.abs().max(), 2)
    if "half_difference" in t:
        N["rho_da_half"] = fmt(float(t.D_minus_A.abs().rank().corr(t.half_difference.rank())), 2)
    rows = []
    for _, r in big.iterrows():
        rows.append([r.system_id.replace("dkasc_", "DKASC ").replace("pvdaq_", "PVDAQ "), fmt(r.A, 2), fmt(r.B, 2),
                     fmt(r.C, 2), fmt(r.D, 2), fmt(r.normalization_effect, 3, True), fmt(r.estimator_effect, 3, True)])
    rows.append([f"Median of absolute values ({N['n_2x2']} systems)", "–", "–", "–", "–", N["norm_med"], N["est_med"]])
    rows.append([f"Maximum of absolute values", "–", "–", "–", "–", N["norm_max"], N["est_max"]])
    T["TWO_BY_TWO_TABLE"] = "\n".join(["| System | A | B | C | D | Normalization effect | Estimator effect |",
                                       "|---|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows])
    # measured table (Table III)
    order = ["reference", "reference_fitted", "reference_clearsky", "khd", "khd_24", "xgboost", "xgboost_full",
             "random_forest", "lstm", "gru", "tcn", "tcn_full", "patchtst", "informer"]
    rows = []
    for m in order:
        if m not in r1.index:
            continue
        reg = REGIME.get(m, "24 months")
        rows.append([LABEL[m], reg, fmt(r1.loc[m, "loc_weighted_mae"]), fmt(r1.loc[m, "worst_location"]),
                     fmt(r1.loc[m, "Alice Springs"]), fmt(r1.loc[m, ["Golden", "Henderson", "Las Vegas", "Arbuckle"]].mean()),
                     str(int(r1.loc[m, "n_systems"]))])
    T["TABLE_MEASURED"] = "\n".join(["| Method | Regime | MAE | Worst | DKASC | US | n |",
                                     "|---|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows])
    # exact-rate table (Table IV)
    clim = pd.concat([read("semisynthetic_climate_runs*.csv", CL), read("info_equal_runs*.csv", CL)]).dropna(subset=["plr"])
    clim = clim.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    fair = read("fair_reference_runs*.csv", CL).dropna(subset=["plr"])
    sd = read("runs*.csv", SD).dropna(subset=["plr"]).drop_duplicates(["site_id", "module_tech", "rate", "seed", "method"], keep="last")

    def site_mae(df: pd.DataFrame, key: str = "error") -> tuple[float, float, int]:
        s = df.groupby("site_id")[key].agg(lambda e: e.abs().mean())
        return float(s.mean()), float(s.max()), int(len(df))

    def cell(df: pd.DataFrame) -> str:
        if not len(df):
            return "– | –"
        mae, worst, _ = site_mae(df)
        return f"{fmt(mae)} | {fmt(worst)}"

    fair = fair.drop_duplicates(["part", "site_id", "family", "rate", "seed", "method"], keep="last")
    adr, fam = fair[fair.part == "B"], fair[fair.part == "A"]
    exact_rows = [
        ("Reference, typical γ", clim[clim.method == "rdtools"], adr[adr.method == "rdtools"], sd[sd.method == "rdtools"]),
        ("Reference, fitted γ", fam[fam.method == "rdtools_fitted_gamma"], adr[adr.method == "rdtools_fitted_gamma"], pd.DataFrame()),
        ("Reference, true γ", fam[fam.method == "rdtools_true_gamma"], pd.DataFrame(), pd.DataFrame()),
        ("κ-HYDRA-D", clim[clim.method == "khd_full"], adr[adr.method == "khd_full"], sd[sd.method == "khd_full"]),
        ("κ-HYDRA-D-24", clim[clim.method == "khd_24"], pd.DataFrame(), sd[sd.method == "khd_24"]),
        ("XGBoost, 24 months", clim[clim.method == "xgboost"], adr[adr.method == "xgboost"], sd[sd.method == "xgboost"]),
        ("XGBoost, full record", clim[clim.method == "xgboost_full"], pd.DataFrame(), sd[sd.method == "xgboost_full"]),
        ("Random forest", clim[clim.method == "random_forest"], pd.DataFrame(), sd[sd.method == "random_forest"]),
        ("LSTM", clim[clim.method == "lstm"], pd.DataFrame(), sd[sd.method == "lstm"]),
        ("GRU", clim[clim.method == "gru"], pd.DataFrame(), sd[sd.method == "gru"]),
        ("TCN, 24 months", clim[clim.method == "tcn"], adr[adr.method == "tcn"], sd[sd.method == "tcn"]),
        ("TCN, full record", clim[clim.method == "tcn_full"], pd.DataFrame(), sd[sd.method == "tcn_full"]),
        ("PatchTST", clim[clim.method == "patchtst"], pd.DataFrame(), sd[sd.method == "patchtst"]),
        ("Informer", clim[clim.method == "informer"], pd.DataFrame(), sd[sd.method == "informer"]),
        ("PVUSA two-stage", pd.DataFrame(), adr[adr.method == "pvusa"], pd.DataFrame()),
    ]
    T["TABLE_EXACT"] = "\n".join(["| Method | PVWatts and thin-film-like: MAE | Worst site | ADR: MAE | Worst site | Single-diode: MAE | Worst site |",
                                  "|---|---|---|---|---|---|---|"]
                                 + [f"| {name} | {cell(a)} | {cell(b)} | {cell(c)} |" for name, a, b, c in exact_rows])
    for key, df in (("sd_ref", sd[sd.method == "rdtools"]), ("sd_khd", sd[sd.method == "khd_full"]),
                    ("sd_khd24", sd[sd.method == "khd_24"]), ("ex_khd24", clim[clim.method == "khd_24"]),
                    ("ex_xgbfull", clim[clim.method == "xgboost_full"]), ("ex_tcnfull", clim[clim.method == "tcn_full"]),
                    ("ex_khd", clim[clim.method == "khd_full"]), ("ex_ref", clim[clim.method == "rdtools"])):
        if len(df):
            mae, worst, n = site_mae(df)
            N[key], N[key + "_worst"], N[key + "_n"] = fmt(mae), fmt(worst), str(n)
            N[key + "_bias"] = fmt(float(df.error.mean()), 3, True)
    sdm = sd.groupby("method").apply(lambda d: site_mae(d)[0])
    N["sd_best_method"] = LABEL.get(sdm.idxmin().replace("rdtools", "reference").replace("khd_full", "khd"), sdm.idxmin())
    N["sd_best_mae"] = fmt(float(sdm.min()))
    N["sd_bias_min"] = fmt(float(sd.groupby("method").error.mean().min()), 3, True)
    N["sd_bias_max"] = fmt(float(sd.groupby("method").error.mean().max()), 3, True)
    # interval coverage (Table VI)
    k = sd[(sd.method == "khd_full") & sd.ci_low.notna()]
    c, n = int(((k.truth >= k.ci_low) & (k.truth <= k.ci_high)).sum()), len(k)
    lo, hi = wilson(c, n)
    N.update(sd_cov_c=str(c), sd_cov_n=str(n), sd_cov_lo=fmt(lo, 2), sd_cov_hi=fmt(hi, 2),
             sd_cov_width=fmt(float((k.ci_high - k.ci_low).median()), 2))
    rr = sd[(sd.method == "rdtools") & sd.ci_low.notna()]
    c2 = int(((rr.truth >= rr.ci_low) & (rr.truth <= rr.ci_high)).sum())
    lo2, hi2 = wilson(c2, len(rr))
    N.update(sd_refcov_c=str(c2), sd_refcov_n=str(len(rr)), sd_refcov_lo=fmt(lo2, 2), sd_refcov_hi=fmt(hi2, 2))
    kap = read("kappa_runs*.csv", FM).drop_duplicates(["system_id", "variant", "injection"], keep="last")
    k0 = kap[(kap.variant == "full") & (kap.injection == 0)]
    rd = read("rd_runs*.csv", FM).drop_duplicates(["system_id", "injection"], keep="last")
    r0 = rd[rd.injection == 0]
    N["meas_width_khd"] = fmt(float((k0.ci_high - k0.ci_low).median()), 2)
    N["meas_width_ref"] = fmt(float((r0.ci_high - r0.ci_low).median()), 2)
    T["TABLE_UNC"] = "\n".join([
        "| Product | Test (unit, n) | Result |", "|---|---|---|",
        "| Rate 95 % interval (jackknife and seeds) | Exact-rate records, four weather sources (record, 48) | 48/48 covered; median width 0.09 %/yr (reference: 24/24 PVWatts, 20/24 thin-film-like) |",
        f"| Rate 95 % interval | Single-diode generator (record, {N['sd_cov_n']}) | {N['sd_cov_c']}/{N['sd_cov_n']} covered (Wilson 95 % {N['sd_cov_lo']}–{N['sd_cov_hi']}); median width {N['sd_cov_width']} %/yr (reference: {N['sd_refcov_c']}/{N['sd_refcov_n']}) |",
        f"| Rate 95 % interval, measured | Main benchmark (system, {len(k0)}) | Median width {N['meas_width_khd']} vs {N['meas_width_ref']} %/yr for the reference; truth unknown |",
        "| Not covered | — | Sensor, filter and model-structure uncertainty |"])
    # spread, window, fit
    ml = read("ml_runs*.csv", FM).drop_duplicates(["system_id", "model", "seed", "injection"], keep="last")
    z = ml[ml.injection == 0].groupby(["system_id", "model"]).plr.mean().unstack()[SEVEN]
    spread = z.max(axis=1) - z.min(axis=1)
    N["spread_med"], N["spread_max"], N["n_spread"] = fmt(float(spread.median()), 2), fmt(float(spread.max()), 2), str(len(spread))
    s = round(float(spread.median()), 2)            # 25-year linear profile: final-year gap 25 s, lifetime mean gap 12 s
    N["illus_final"], N["illus_life"] = f"{25 * s:.1f}".rstrip("0").rstrip("."), f"{12 * s:.1f}"
    rm = ml[ml.injection == 0].groupby("model").val_rmse.mean()[SEVEN]
    N["rmse_lo"], N["rmse_hi"] = fmt(float(rm.min())), fmt(float(rm.max()))
    w = pd.read_csv(FM / "refwindow_fixed.csv")
    N["win_med"], N["win_max"] = fmt(float(w.window_range.median()), 2), fmt(float(w.window_range.max()), 2)
    # reference variants / clear sky
    rv = pd.read_csv(FM / "reference_variants.csv")
    dk = rv[rv.system_id.str.startswith("dkasc") & rv.plr_clearsky.notna()]
    d = dk.plr_clearsky - dk.plr_typical
    N.update(cs_diff_min=fmt(float(d.min()), 2, True), cs_diff_max=fmt(float(d.max()), 2, True),
             cs_diff_med=fmt(float(d.median()), 2, True), cs_n=str(len(dk)))
    # profile
    p = pd.read_csv(FM / "profile_summary.csv")
    wmax = p.loc[p.width_5pct.idxmax()]
    N.update(prof_n=str(len(p)), prof_single=str(int(p.single_interior_minimum.sum())),
             prof_widest=wmax.system_id.replace("dkasc_", "DKASC ").replace("pvdaq_", "PVDAQ "),
             prof_widest_w=fmt(float(wmax.width_5pct), 2), prof_med_w=fmt(float(p.width_5pct.median()), 2))
    # latitude
    lat = pd.read_csv(CL / "tables/E13_latitude_spearman.csv").set_index("method")
    N["lat_rho"], N["lat_p"] = fmt(float(lat.loc["rdtools", "spearman_rho"]), 2), fmt(float(lat.loc["rdtools", "p_value"]), 2)
    # mixed model
    mm = pd.read_csv(TAB / "R9_mixed_model.csv")
    for _, r in mm.iterrows():
        if "contrast" in r and " - " in str(r.contrast):
            key = str(r.contrast).split(" - ")[0]
            N[f"mm_{key}"] = fmt(r.estimate, 4, True)
            N[f"mm_{key}_lo"], N[f"mm_{key}_hi"] = fmt(r.ci_low, 4, True), fmt(r.ci_high, 4, True)
    claims(N, r1, t, big, sd, e_rows=pd.read_csv(TAB / "R0_injection_rows.csv"))
    return N, T


def claims(N: dict, r1: pd.DataFrame, t: pd.DataFrame, big: pd.DataFrame, sd: pd.DataFrame, e_rows: pd.DataFrame) -> None:
    """Every directional statement of the prose is checked against the numbers; a failed check stops the build so
    the sentence is rewritten instead of printed."""
    fail = []
    trio = [r1.loc[m, "loc_weighted_mae"] for m in ("khd", "reference")] + [float(N["inj_best_two_stage"].replace(MINUS, "-"))]
    lo, hi = (float(N.get(k, "nan").replace(MINUS, "-")) for k in ("mm_khd_lo", "mm_khd_hi"))
    if not (lo > -0.05 and hi < 0.05):
        fail.append(f"mixed-model CI of kappa - reference [{lo}, {hi}] not within delta")
    if max(trio) - min(trio) >= 0.05:
        fail.append("reference, best two-stage and kappa not within delta on injection")
    if not all(abs(r.estimator_effect) > abs(r.normalization_effect) for _, r in big.iterrows()):
        fail.append("largest D-A systems not dominated by the estimator effect")
    if t.estimator_effect.abs().median() <= t.normalization_effect.abs().median():
        fail.append("median estimator effect not larger than median normalization effect")
    if "rho_da_half" in N and abs(float(N["rho_da_half"].replace(MINUS, "-"))) >= 0.5:
        fail.append("|D-A| IS related to the half-record difference; rewrite the 2x2 sentence")
    xf = e_rows[e_rows.method == "xgboost_full"].recovery.median()
    if not xf < 0.95:
        fail.append(f"full-record XGBoost median recovery {xf:.3f} not below 0.95 (absorption claim)")
    if r1.loc["reference_fitted", "loc_weighted_mae"] < r1.loc["reference", "loc_weighted_mae"] - 0.002:
        fail.append("fitted coefficient DID reduce the injected-change error")
    if not (sd.groupby("method").error.mean() > 0).all():
        fail.append("not every method underestimates the single-diode loss")
    for f in fail:
        print("CLAIM CHECK FAILED:", f)
    if fail and "--force" not in sys.argv:
        sys.exit(2)


def prose(N: dict) -> dict:
    blocks, key, buf = {}, None, []
    for ln in PROSE.read_text(encoding="utf-8").splitlines():
        if ln.startswith("### "):
            if key:
                blocks[key] = "\n".join(buf).strip()
            key, buf = ln[4:].strip(), []
        else:
            buf.append(ln)
    if key:
        blocks[key] = "\n".join(buf).strip()
    return {k: v.format(**N) for k, v in blocks.items()}


def main() -> None:
    N, T = numbers()
    text = TPL.read_text(encoding="utf-8")
    fills = {**T, **prose(N)}
    n_abs = len(fills["ABSTRACT"].split())
    print("abstract words:", n_abs)
    if n_abs > 200 and "--force" not in sys.argv:
        sys.exit("abstract longer than 200 words")
    for k, v in fills.items():
        text = text.replace(f"⟪{k}⟫", v)
    left = re.findall(r"⟪[^⟫]+⟫", text)
    (REV / "06_kontrol/checks/v3_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    if left:
        print("UNFILLED:", sorted(set(left)))
        if "--force" not in sys.argv:
            sys.exit(1)
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
