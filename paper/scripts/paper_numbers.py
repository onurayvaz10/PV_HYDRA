"""Numbers, tables and claim checks of the paper from the result files (first-year rate convention throughout).

The manuscript template (not shipped) holds «name» number fields and ⟪TABLE⟫ blocks. Every number is computed here from
the run files; every directional sentence is checked (claims); a failed check stops the build so the sentence is
rewritten instead of printed. build_result_ledger.py calls numbers() to recompute the ledger.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
sys.path.insert(0, str(ROOT))
from src.degradation.truth_convention import first_year  # noqa: E402

FM, TAB = ROOT / "results/fixed_mask", ROOT / "results/fixed_mask/tables"
SD, CL, FY = ROOT / "results/singlediode", ROOT / "results/climate_expansion", ROOT / "results/first_year"
TPL, OUT = REV / "manuscript_template.md", REV / "manuscript.md"            # template not shipped
MINUS, DELTA = "−", 0.05
SEVEN = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
NAME = {"xgboost": "XGBoost", "random_forest": "Random forest", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN",
        "patchtst": "PatchTST", "informer": "Informer", "xgboost_w24": "XGBoost, 24-h input", "rdtools": "Reference",
        "reference": "Reference", "khd": "κ-HYDRA-D", "khd_full": "κ-HYDRA-D", "khd_24": "κ-HYDRA-D-24",
        "khd_linear": "κ-HYDRA-D, linear correction", "xgboost_full": "XGBoost, full record", "tcn_full": "TCN, full record"}


def fmt(x, nd=3, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", MINUS)


def num(s: str) -> float:
    return float(s.replace(MINUS, "-"))


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


def site_mae(df: pd.DataFrame) -> tuple[float, float]:
    s = df.groupby("site_id").err.apply(lambda e: e.abs().mean())
    return float(s.mean()), float(s.max())


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def numbers() -> tuple[dict, dict, dict]:
    N, T, D = {}, {}, {}
    # ---------------------------------------------------------------- measured injection (Eq. 2 convention)
    r1 = pd.read_csv(TAB / "R1_injection_by_method.csv").set_index("method")
    for m in r1.index:
        N[f"inj_{m}"] = fmt(r1.loc[m, "loc_weighted_mae"])
        N[f"injworst_{m}"] = fmt(r1.loc[m, "worst_location"])
    best2 = min([m for m in SEVEN if m in r1.index], key=lambda m: r1.loc[m, "loc_weighted_mae"])
    N["best_two_stage"], N["inj_best_two_stage"] = NAME[best2], fmt(r1.loc[best2, "loc_weighted_mae"])
    arb = pd.read_csv(TAB / "R2_arbuckle_recovery.csv").set_index("method")
    for m in ("khd", "reference", "khd_24"):
        if m in arb.index:
            N[f"arb_{m}_min"], N[f"arb_{m}_max"] = fmt(arb.loc[m].min(), 2), fmt(arb.loc[m].max(), 2)
    off = pd.read_csv(TAB / "R2c_arbuckle_clipoff.csv")
    off = off[(off["mask"] == "clip off") & (off.coefficient == "typical")].drop(columns=["mask", "coefficient"])
    N["arb_off_min"], N["arb_off_max"] = fmt(float(off.min(axis=1).iloc[0]), 2), fmt(float(off.max(axis=1).iloc[0]), 2)
    lolo = pd.read_csv(TAB / "R12_leave_one_location_out.csv").set_index("method")
    for m in ("khd", "reference"):
        v = lolo.loc[m].drop("all locations")
        N[f"lolo_{m}_min"], N[f"lolo_{m}_max"] = fmt(v.min()), fmt(v.max())
    mm = pd.read_csv(TAB / "R9_mixed_model.csv")
    for _, r in mm.iterrows():
        if " - " in str(r.contrast):
            k = str(r.contrast).split(" - ")[0]
            N[f"mm_{k}"], N[f"mm_{k}_lo"], N[f"mm_{k}_hi"] = fmt(r.estimate, 4, True), fmt(r.ci_low, 4, True), fmt(r.ci_high, 4, True)
    # ---------------------------------------------------------------- known-rate records (first-year truth)
    clim = pd.concat([read("semisynthetic_climate_runs*.csv", CL), read("info_equal_runs*.csv", CL)]).dropna(subset=["plr"])
    clim = clim.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    clim["err"] = clim.plr - first_year(clim.rate)
    fair = read("fair_reference_runs*.csv", CL).dropna(subset=["plr"]).drop_duplicates(
        ["part", "site_id", "family", "rate", "seed", "method"], keep="last")
    fair["err"] = fair.plr - first_year(fair.truth)
    sd = read("runs*.csv", SD).dropna(subset=["plr"]).drop_duplicates(["site_id", "module_tech", "rate", "seed", "method"], keep="last")
    sd["truth"] = first_year(sd.truth)
    sd["err"] = sd.plr - sd.truth
    D.update(clim=clim, fair=fair, sd=sd)
    ex = {m: site_mae(g) for m, g in clim.groupby("method")}
    sdm = {m: site_mae(g) for m, g in sd.groupby("method")}
    for m, (a, w) in ex.items():
        N[f"ex_{m}"], N[f"exworst_{m}"] = fmt(a), fmt(w)
    for m, (a, w) in sdm.items():
        N[f"sd_{m}"], N[f"sdworst_{m}"] = fmt(a), fmt(w)
    ref_sites = clim[clim.method == "rdtools"].groupby("site_id").err.apply(lambda e: e.abs().mean())
    N["ex_ref_worst_site"] = ref_sites.idxmax().replace("_", " ").title()
    adr = fair[fair.part == "B"]
    fam = fair[fair.part == "A"]
    for m, g in fam.groupby("method"):
        N[f"exA_{m}"], N[f"exAworst_{m}"] = (fmt(x) for x in site_mae(g))
    for m, g in adr.groupby("method"):
        N[f"adr_{m}"], N[f"adrworst_{m}"] = (fmt(x) for x in site_mae(g))
    bias = sd.groupby("method").err.mean()
    N["sd_bias_min"], N["sd_bias_max"] = fmt(bias.min(), 3, True), fmt(bias.max(), 3, True)
    best_sd = min(sdm, key=lambda m: sdm[m][0])
    N["sd_best_method"], N["sd_best_mae"] = NAME.get(best_sd, best_sd), fmt(sdm[best_sd][0])
    N["conv_m1"], N["conv_m05"] = fmt(float(first_year(-1.0)), 4), fmt(float(first_year(-0.5)), 4)
    # ---------------------------------------------------------------- power fit versus rate accuracy
    fit = pd.read_csv(TAB / "R13_fit_vs_rate.csv").set_index("model")
    seven = fit.loc[[m for m in SEVEN if m in fit.index]]
    N["rmse_lo"], N["rmse_hi"] = fmt(seven.val_rmse.min()), fmt(seven.val_rmse.max())
    N["r2_lo"], N["r2_hi"] = fmt(seven.val_r2.min()), fmt(seven.val_r2.max())
    lo_fit = seven.val_rmse.idxmin()
    N["lowest_rmse_model"], N["lowest_rmse"] = NAME[lo_fit], fmt(seven.loc[lo_fit, "val_rmse"], 4)
    N["lowest_rmse_inj"] = fmt(r1.loc[lo_fit, "loc_weighted_mae"])
    N["lowest_rmse_inj_rank"] = str(int(r1.loc[SEVEN, "loc_weighted_mae"].rank().loc[lo_fit]))
    N["best_two_stage_rmse"] = fmt(seven.loc[best2, "val_rmse"], 4)
    N["best_two_stage_rmse_rank"] = str(int(seven.val_rmse.rank().loc[best2]))
    rho = spearmanr(seven.val_rmse, r1.loc[seven.index, "loc_weighted_mae"]).statistic
    N["rho_fit_inj"] = fmt(rho, 2)
    rows = []
    for m in SEVEN + ["xgboost_w24"]:
        if m in fit.index:
            rows.append([NAME[m], fmt(fit.loc[m, "val_rmse"], 4), fmt(fit.loc[m, "val_r2"]),
                         N.get(f"inj_{m}", "–"), N.get(f"ex_{m}", "–"), N.get(f"sd_{m}", "–")])
    T["TABLE_FIT"] = table(["Model", "Fit RMSE", "Fit R²", "Injected change", "Exact rate", "Single-diode"], rows)
    # ---------------------------------------------------------------- spread, window, 2 x 2, profile
    ml = read("ml_runs*.csv", FM).drop_duplicates(["system_id", "model", "seed", "injection"], keep="last")
    z = ml[ml.injection == 0].groupby(["system_id", "model"]).plr.mean().unstack()[SEVEN]
    spread = z.max(axis=1) - z.min(axis=1)
    N["spread_med"], N["spread_max"] = fmt(spread.median(), 2), fmt(spread.max(), 2)
    s = round(float(spread.median()), 2)
    N["illus_final"], N["illus_life"] = f"{25 * s:.1f}".rstrip("0").rstrip("."), f"{12 * s:.1f}"
    w = pd.read_csv(FM / "refwindow_fixed.csv")
    N["win_med"], N["win_max"] = fmt(w.window_range.median(), 2), fmt(w.window_range.max(), 2)
    t = pd.read_csv(TAB / "R3_two_by_two.csv")
    N["norm_med"], N["norm_max"] = fmt(t.normalization_effect.abs().median()), fmt(t.normalization_effect.abs().max())
    N["est_med"], N["est_max"] = fmt(t.estimator_effect.abs().median()), fmt(t.estimator_effect.abs().max())
    big = t.loc[t.D_minus_A.abs().sort_values(ascending=False).index[:3]]
    N["da_largest_systems"] = ", ".join(x.replace("dkasc_", "DKASC ").replace("pvdaq_", "PVDAQ ") for x in big.system_id)
    N["da_largest_min"], N["da_largest_max"] = fmt(big.D_minus_A.abs().min(), 2), fmt(big.D_minus_A.abs().max(), 2)
    N["rho_da_half"] = fmt(float(t.D_minus_A.abs().rank().corr(t.half_difference.rank())), 2)
    rows = [[x.system_id.replace("dkasc_", "DKASC ").replace("pvdaq_", "PVDAQ "), fmt(x.A, 2), fmt(x.B, 2), fmt(x.C, 2),
             fmt(x.D, 2), fmt(x.normalization_effect, 3, True), fmt(x.estimator_effect, 3, True)] for _, x in big.iterrows()]
    rows += [[f"Median of absolute values ({len(t)} systems)", "–", "–", "–", "–", N["norm_med"], N["est_med"]],
             ["Maximum of absolute values", "–", "–", "–", "–", N["norm_max"], N["est_max"]]]
    T["TABLE_2X2"] = table(["System", "A", "B", "C", "D", "Normalization effect", "Estimator effect"], rows)
    p = pd.read_csv(FM / "profile_summary.csv")
    wmax = p.loc[p.width_5pct.idxmax()]
    N.update(prof_n=str(len(p)), prof_single=str(int(p.single_interior_minimum.sum())),
             prof_widest=wmax.system_id.replace("dkasc_", "DKASC ").replace("pvdaq_", "PVDAQ "),
             prof_widest_w=fmt(wmax.width_5pct, 2), prof_med_w=fmt(p.width_5pct.median(), 2))
    rv = pd.read_csv(FM / "reference_variants.csv")
    dk = rv[rv.system_id.str.startswith("dkasc") & rv.plr_clearsky.notna()]
    dd = dk.plr_clearsky - dk.plr_typical
    N.update(cs_diff_min=fmt(dd.min(), 2, True), cs_diff_max=fmt(dd.max(), 2, True), cs_diff_med=fmt(dd.median(), 2, True))
    lat = pd.read_csv(CL / "tables/E13_latitude_spearman.csv").set_index("method")
    N["lat_rho"], N["lat_p"] = fmt(lat.loc["rdtools", "spearman_rho"], 2), fmt(lat.loc["rdtools", "p_value"], 2)
    # ---------------------------------------------------------------- measured information regimes (Table)
    regime = {"reference": "–", "reference_fitted": "first 12 months", "reference_clearsky": "–", "khd": "full record",
              "khd_24": "24 months", "khd_linear": "full record", "xgboost": "24 months", "xgboost_w24": "24 months",
              "xgboost_full": "full record", "tcn": "24 months", "tcn_full": "full record"}
    label = {"reference": "Reference, typical γ", "reference_fitted": "Reference, fitted γ", "reference_clearsky": "Reference, clear-sky",
             "khd": "κ-HYDRA-D", "khd_24": "κ-HYDRA-D-24", "khd_linear": "κ-HYDRA-D, linear correction",
             "xgboost": "XGBoost", "xgboost_w24": "XGBoost, 24-h input", "xgboost_full": "XGBoost", "tcn": "TCN", "tcn_full": "TCN"}
    rows = []
    for m in label:
        if m in r1.index:
            rows.append([label[m], regime[m], fmt(r1.loc[m, "loc_weighted_mae"]), fmt(r1.loc[m, "worst_location"]),
                         fmt(r1.loc[m, "Alice Springs"]), str(int(r1.loc[m, "n_systems"]))])
    T["TABLE_REGIME"] = table(["Method", "Regime", "MAE", "Worst location", "DKASC", "Systems"], rows)
    # ---------------------------------------------------------------- exact-rate table (three generators)
    def cell(df):
        if not len(df):
            return ["–", "–"]
        a, w_ = site_mae(df)
        return [fmt(a), fmt(w_)]
    ex_rows = [("Reference, typical γ", clim[clim.method == "rdtools"], adr[adr.method == "rdtools"], sd[sd.method == "rdtools"]),
               ("Reference, fitted γ", fam[fam.method == "rdtools_fitted_gamma"], adr[adr.method == "rdtools_fitted_gamma"], sd.iloc[0:0]),
               ("Reference, true γ", fam[fam.method == "rdtools_true_gamma"], adr.iloc[0:0], sd.iloc[0:0]),
               ("κ-HYDRA-D", clim[clim.method == "khd_full"], adr[adr.method == "khd_full"], sd[sd.method == "khd_full"]),
               ("κ-HYDRA-D-24", clim[clim.method == "khd_24"], adr.iloc[0:0], sd[sd.method == "khd_24"]),
               ("κ-HYDRA-D, linear correction", clim[clim.method == "khd_linear"], adr.iloc[0:0], sd.iloc[0:0])]
    for m in SEVEN + ["xgboost_w24", "xgboost_full", "tcn_full"]:
        ex_rows.append((NAME[m], clim[clim.method == m], adr[adr.method == m], sd[sd.method == m]))
    ex_rows.append(("PVUSA two-stage", clim.iloc[0:0], adr[adr.method == "pvusa"], sd.iloc[0:0]))
    T["TABLE_EXACT"] = table(["Method", "PVWatts and thin-film-like: MAE", "Worst site", "ADR: MAE", "Worst site",
                              "Single-diode: MAE", "Worst site"],
                             [[n] + cell(a) + cell(b) + cell(c) for n, a, b, c in ex_rows if len(a) or len(b) or len(c)])
    # ---------------------------------------------------------------- ablation and violations (first-year truth)
    ab = pd.read_csv(FY / "FY_ablation.csv").set_index(["set", "case", "variant"])
    def abv(st, case, v):
        return fmt(ab.loc[(st, case, v), "mae"]) if (st, case, v) in ab.index else "–"
    adr_v = {m: site_mae(adr[adr.method == m])[0] for m in ("khd_full", "khd_no_physics", "khd_no_correction", "rdtools")}
    rows = [["Thin-film-like, long (24)"] + [abv("semi", "thin_film", v) for v in ("full", "no_physics", "no_correction", "rdtools")],
            ["Thin-film-like, first 3 years (12)"] + [abv("short", "thin_film", v) for v in ("full", "no_physics", "no_correction", "rdtools")],
            ["ADR response, 20 weather sites (120)"] + [fmt(adr_v[m]) for m in ("khd_full", "khd_no_physics", "khd_no_correction", "rdtools")],
            ["Unobserved soiling (12)"] + [abv("hard", "soiling", v) for v in ("full", "no_physics", "no_correction", "rdtools")],
            ["3 % step (12)"] + [abv("hard", "step", v) for v in ("full", "no_physics", "no_correction", "rdtools")]]
    T["TABLE_ABLATION"] = table(["Scenario (n)", "Full", "No physics", "No correction", "Reference"], rows)
    hard = pd.read_csv(FY / "FY_hard.csv").set_index(["scenario", "method"])
    N["cp_step"] = fmt(hard.loc[("step", "changepoint"), "mae"])
    N["short_tf_nophys_p"] = fmt(ab.loc[("short", "thin_film", "no_physics"), "p_wilcoxon"], 2)
    N["short_tf_full"], N["short_tf_nophys"] = abv("short", "thin_film", "full"), abv("short", "thin_film", "no_physics")
    # ---------------------------------------------------------------- intervals
    cov = pd.read_csv(FY / "FY_coverage_4site.csv").set_index(["method", "family"])
    k48 = int(cov.loc["khd_full"].covered.sum())
    N["cov48_khd"] = f"{k48}/48"
    N["cov48_ref_pv"] = f"{int(cov.loc[('rdtools', 'pvwatts'), 'covered'])}/24"
    N["cov48_ref_tf"] = f"{int(cov.loc[('rdtools', 'thin_film'), 'covered'])}/24"
    N["cov48_xgb"] = f"{int(cov.loc['xgboost'].covered.sum())}/48"
    N["cov48_width"] = fmt(cov.loc["khd_full"].median_width.mean(), 2)
    kk = sd[(sd.method == "khd_full") & sd.ci_low.notna()]
    c = int(((kk.truth >= kk.ci_low) & (kk.truth <= kk.ci_high)).sum())
    lo, hi = wilson(c, len(kk))
    N.update(sd_cov_c=str(c), sd_cov_n=str(len(kk)), sd_cov_lo=fmt(lo, 2), sd_cov_hi=fmt(hi, 2),
             sd_cov_width=fmt((kk.ci_high - kk.ci_low).median(), 2))
    rr = sd[(sd.method == "rdtools") & sd.ci_low.notna()]
    c2 = int(((rr.truth >= rr.ci_low) & (rr.truth <= rr.ci_high)).sum())
    N.update(sd_refcov_c=str(c2), sd_refcov_n=str(len(rr)))
    kap = read("kappa_runs*.csv", FM).drop_duplicates(["system_id", "variant", "injection"], keep="last")
    k0 = kap[(kap.variant == "full") & (kap.injection == 0)]
    rd = read("rd_runs*.csv", FM).drop_duplicates(["system_id", "injection"], keep="last")
    r0 = rd[rd.injection == 0]
    N["meas_width_khd"], N["meas_width_ref"] = fmt((k0.ci_high - k0.ci_low).median(), 2), fmt((r0.ci_high - r0.ci_low).median(), 2)
    bb = pd.read_csv(TAB / "R11_block_bootstrap.csv")
    N["bb_width_def"], N["bb_width_block"] = fmt(bb.width_default.median(), 2), fmt(bb.width_block.median(), 2)
    N["bb_ratio"] = fmt(bb.width_ratio.median(), 1)
    T["TABLE_UNC"] = table(["Product", "Test (unit, n)", "Result"], [
        ["Rate 95 % interval (jackknife and seeds)", "Exact-rate records, four weather sources (record, 48)",
         f"{N['cov48_khd']} covered; median width {N['cov48_width']} %/yr (reference: {N['cov48_ref_pv']} PVWatts, {N['cov48_ref_tf']} thin-film-like; XGBoost {N['cov48_xgb']})"],
        ["Rate 95 % interval", f"Single-diode generator (record, {N['sd_cov_n']})",
         f"{N['sd_cov_c']}/{N['sd_cov_n']} covered (Wilson 95 % {N['sd_cov_lo']}–{N['sd_cov_hi']}); reference {N['sd_refcov_c']}/{N['sd_refcov_n']}"],
        ["Rate 95 % interval, measured", "Main benchmark (system, 18)",
         f"Median width {N['meas_width_khd']} vs {N['meas_width_ref']} %/yr for the reference; truth unknown"],
        ["Reference interval, serial dependence", "Main benchmark (system, 18)",
         f"Median width {N['bb_width_def']} (default) vs {N['bb_width_block']} %/yr (30-day block bootstrap)"],
        ["Not covered", "—", "Sensor, filter and model-structure uncertainty"]])
    # ---------------------------------------------------------------- controls
    fitc = fit
    if "xgboost_w24" in fitc.index and "xgboost_w24" in r1.index:
        N["w24_rmse"], N["xgb_rmse"] = fmt(fitc.loc["xgboost_w24", "val_rmse"], 4), fmt(fitc.loc["xgboost", "val_rmse"], 4)
        dinj = r1.loc["xgboost_w24", "loc_weighted_mae"] - r1.loc["xgboost", "loc_weighted_mae"]
        dex = ex["xgboost_w24"][0] - ex["xgboost"][0] if "xgboost_w24" in ex else np.nan
        closer = fitc.loc["xgboost_w24", "val_rmse"] < fitc.loc["xgboost", "val_rmse"]
        N["w24_closer_worse"] = "1" if closer and dinj > 0 else "0"
        N["w24_sentence"] = (
            f"With the same 24-h input as the sequence models, XGBoost fitted power {'more' if closer else 'less'} closely "
            f"(RMSE {N['w24_rmse']} vs {N['xgb_rmse']}) and erred by {N['inj_xgboost_w24']} %/yr on injected records and "
            f"{N.get('ex_xgboost_w24', '–')} %/yr on exact-rate records ({N['inj_xgboost']} and {N['ex_xgboost']} %/yr with the "
            "3-h input); "
            + ("the closer fit thus came with a larger rate error, " if closer and dinj > 0 else "")
            + ("a change below δ, so input length does not explain the ranking of the models."
               if abs(dinj) < DELTA and (not np.isfinite(dex) or abs(dex) < DELTA) else
               "a change above δ, so input length alone can move the rate error."))
    if "khd_linear" in r1.index and "khd_linear" in ex:
        tf = clim[clim.family == "thin_film"]
        tf_lin, tf_mlp = site_mae(tf[tf.method == "khd_linear"])[0], site_mae(tf[tf.method == "khd_full"])[0]
        tf_ref = site_mae(tf[tf.method == "rdtools"])[0]
        N.update(lin_tf=fmt(tf_lin), mlp_tf=fmt(tf_mlp), ref_tf=fmt(tf_ref))
        N["lin_sentence"] = (
            f"A linear correction with the same inputs and joint trend erred by {N['inj_khd_linear']} %/yr on injected records "
            f"and {N['ex_khd_linear']} %/yr on exact-rate records ({N['inj_khd']} and {N['ex_khd_full']} %/yr for the two-layer "
            f"correction); on the thin-film-like records, where the PVWatts response is misspecified, it erred by {N['lin_tf']} "
            f"%/yr against {N['mlp_tf']} %/yr for the two-layer correction and {N['ref_tf']} %/yr for the reference. "
            + ("The non-linear correction therefore added little beyond a linear one with a joint trend."
               if tf_lin - tf_mlp < DELTA / 5 else
               "The non-linear correction therefore mattered where the response was misspecified."))
        N["lin_discussion"] = (
            "A linear correction with the same inputs and joint trend was about as accurate as the two-layer network, so the "
            "gain lies in the joint, time-blind correction rather than in network depth."
            if tf_lin - tf_mlp < DELTA / 5 else
            "A linear correction with the same inputs and joint trend was less accurate on the misspecified responses, so the "
            "flexibility of the correction contributes beyond the joint trend.")
    return N, T, D


def claims(N: dict, D: dict) -> list[str]:
    fail = []
    trio = [num(N["inj_khd"]), num(N["inj_reference"]), num(N["inj_best_two_stage"])]
    if max(trio) - min(trio) >= DELTA:
        fail.append("reference, best two-stage and kappa not within delta on injection")
    if not (num(N["mm_khd_lo"]) > -DELTA and num(N["mm_khd_hi"]) < DELTA):
        fail.append("mixed-model CI not within delta")
    if int(N["lowest_rmse_inj_rank"]) <= 3:
        fail.append("lowest-RMSE model is among the three most accurate in rate: fit-vs-rate claim not supported")
    if int(N["best_two_stage_rmse_rank"]) <= 2:
        fail.append("most rate-accurate two-stage model is among the two best fits")
    if num(N["est_med"]) <= num(N["norm_med"]):
        fail.append("median estimator effect not larger than normalization effect")
    if abs(num(N["rho_da_half"])) >= 0.5:
        fail.append("|D-A| related to half-record difference; rewrite")
    if not (D["sd"].groupby("method").err.mean() > 0).all():
        fail.append("not every method underestimates the single-diode loss")
    if num(N["inj_xgboost_full"]) <= num(N["inj_xgboost"]) + DELTA:
        fail.append("full-record XGBoost not worse by more than delta (absorption sentence)")
    if N["ex_ref_worst_site"] != "Utrecht":
        fail.append(f"discussion names Utrecht as the worst reference site, data say {N['ex_ref_worst_site']}")
    if num(N["exA_rdtools_fitted_gamma"]) > 0.6 * num(N["ex_rdtools"]) or num(N["exA_rdtools_fitted_gamma"]) < 0.4 * num(N["ex_rdtools"]):
        fail.append("discussion says the fitted coefficient removed about half of the error; not supported")
    if num(N["lat_p"]) >= 0.05 or num(N["lat_rho"]) <= 0:
        fail.append("latitude sentence (grew with latitude) not supported")
    if N["lowest_rmse_inj_rank"] != "7":
        fail.append("abstract says the closest fit gave the largest injected-change error, but it is not last")
    if N.get("w24_closer_worse") != "1":
        fail.append("discussion says a closer fit with a longer input worsened the rate; not supported")
    for k in ("w24_sentence", "lin_sentence"):
        if k not in N:
            fail.append(f"{k} missing: control runs incomplete")
    return fail


def main() -> None:
    N, T, D = numbers()
    fail = claims(N, D)
    text = TPL.read_text(encoding="utf-8")
    for k, v in T.items():
        text = text.replace(f"⟪{k}⟫", v)
    missing = sorted(set(re.findall(r"«([\w]+)»", text)) - set(N))
    text = re.sub(r"«([\w]+)»", lambda m: N.get(m.group(1), m.group(0)), text)
    left = re.findall(r"⟪[^⟫]+⟫|«[\w]+»", text)
    abstract = re.search(r"^%% abstract: (.+)$", text, re.M).group(1)
    n_abs = len(abstract.split())
    (REV / "paper_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print("abstract words:", n_abs)
    for f in fail:
        print("CLAIM CHECK FAILED:", f)
    if missing or left:
        print("UNFILLED:", missing or left)
    if n_abs > 250:
        print("ABSTRACT TOO LONG")
    if (fail or missing or left or n_abs > 250) and "--force" not in sys.argv:
        sys.exit(2)
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
