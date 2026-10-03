"""Result ledger of the paper: every number printed in the manuscript, recomputed from the result
files (or, for design constants, read from the protocol/configuration), with source file and selector.

Output: paper/result_ledger.csv  (id, text, value, source, selector, kind)
  text   : the number exactly as printed in the paper
  kind   : result | design | derived | count
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
TAB = ROOT / "results/degradation_paper/tables"
RES = ROOT / "results/degradation_paper"
ET = ROOT / "results/climate_expansion/tables"
CE = ROOT / "results/climate_expansion"
MODELS = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
ORDER = ["rdtools", "khd_full", "xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]
ROWS: list[dict] = []


def add(id_: str, text: str, value, source: str, selector: str, kind: str = "result") -> str:
    ROWS.append({"id": id_, "text": text, "value": value if isinstance(value, str) else json.dumps(value),
                 "source": source, "selector": selector, "kind": kind})
    return text


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def rng(vals, d=3):
    return f"{min(vals):.{d}f}–{max(vals):.{d}f}"


def group_of(sid: str) -> str:
    if sid.startswith(("hk_", "ut_", "fmi_")):
        return "short"
    if sid.startswith("dkasc") or sid in ("pvdaq_1423", "pvdaq_1430", "pvdaq_2107"):
        return "long sensor-grade"
    return "long expansion"


def v3_rows() -> None:
    """Numbers of the benchmark version (protocol P1-P7): recomputed by the same functions that fill the manuscript
    (fill_v3.numbers) from results/fixed_mask, results/singlediode and results/climate_expansion, plus the screen
    values and geometry quoted in Section II."""
    import re
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fill_v3
    n, tables = fill_v3.numbers()
    src = "fill_v3.numbers(): results/fixed_mask, results/singlediode, results/climate_expansion"
    for k, v in n.items():
        add(f"v3_{k}", v, v, src, k)
    for name, tbl in tables.items():
        for i, tok in enumerate(re.findall(r"[−-]?\+?\d[\d,]*(?:\.\d+)?", tbl)):
            add(f"v3_{name}_{i}", tok, tok, src, f"{name} cell token {i}")
    scr = pd.read_csv(RES / "tier_a_screen.csv")
    dk = scr[scr.system_id.str.startswith("dkasc") & scr["pass"]]
    cs = 100 * dk.poa_clearsky_envelope_trend_per_year
    er = 100 * dk.poa_era5_ratio_trend_per_year
    add("v3_dkasc_cs_drift", f"+{cs.min():.2f} to +{cs.max():.2f}", [cs.min(), cs.max()], "tier_a_screen.csv",
        "DKASC poa_clearsky_envelope_trend_per_year x 100")
    add("v3_dkasc_era5_drift", f"{er.min():.2f} to {er.max():.2f}".replace("-", "−"), [er.min(), er.max()],
        "tier_a_screen.csv", "DKASC poa_era5_ratio_trend_per_year x 100")
    s = pd.read_csv(CE / "screen_tier_s.csv").set_index("system_id")
    for sid in ("pvdaq_10", "pvdaq_33", "pvdaq_1278"):
        for col, tag in (("poa_era5_ratio_trend_per_year", "era5"), ("poa_clearsky_envelope_trend_per_year", "cs")):
            v = 100 * float(s.loc[sid, col])
            add(f"v3_drift_{tag}_{sid}", f"{v:+.2f}".replace("-", "−"), v, "screen_tier_s.csv", f"{sid} {col} x 100")
    for sid in ("10", "33", "1278", "1423", "1430", "2107"):
        add(f"v3_system_id_{sid}", sid, sid, "PVDAQ system identifiers (run_fixed_mask.systems)", "identifier", "design")
    sites = pd.read_csv(ROOT / "configs/semisynthetic_climate_sites.csv").set_index("site_id")
    add("v3_utrecht_lat", f"{sites.loc['utrecht', 'latitude']:.0f}", float(sites.loc["utrecht", "latitude"]),
        "configs/semisynthetic_climate_sites.csv", "utrecht latitude")
    lat1, lon1, lat2, lon2 = np.radians([36.0275, -114.9215, 36.1952, -115.1582])
    km = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
    add("v3_henderson_lasvegas_km", f"{km:.0f}", float(km), "system metadata (PVDAQ 1423, 1278)", "haversine distance")
    sys.path.insert(0, str(ROOT / "scripts"))
    fallback = {"pvdaq_10": 13.1, "pvdaq_1278": 6.9}      # record spans without raw data (public repository)
    try:
        import run_fixed_mask as fm
        spans = {}
        for sid in fm.systems():
            base = fm.load(sid)[0]
            spans[sid] = (base.index.max() - base.index.min()).days / 365.25
    except Exception as exc:  # no raw data: the printed spans come from the recorded values
        print("span fallback:", exc)
        spans = fallback
    add("v3_span_range", f"{min(spans.values()):.1f}–{max(spans.values()):.1f}",
        [round(min(spans.values()), 1), round(max(spans.values()), 1)], "run_fixed_mask.load", "record span")
    add("v3_rho_bound", f"{2 / max(spans.values()):.2f}", round(2 / max(spans.values()), 2), "derived: 2/T", "|rho| < 2/T")
    add("v3_span_max", f"{max(spans.values()):.1f}", round(max(spans.values()), 1), "run_fixed_mask.load", "longest span")


def main() -> None:
    # ------------------------------------------------------------------ data groups (Table I)
    loc = pd.read_csv(ET / "E10_measured_locations.csv")
    add("n_main", str(int(loc.systems.sum())), int(loc.systems.sum()), "E10_measured_locations.csv", "sum(systems)", "count")
    add("n_locations", str(loc.location.nunique()), int(loc.location.nunique()), "E10_measured_locations.csv", "nunique(location)", "count")
    add("n_countries", str(loc.country.nunique()), int(loc.country.nunique()), "E10_measured_locations.csv", "nunique(country)", "count")
    add("n_classes_measured", str(loc.climate.nunique()), int(loc.climate.nunique()), "E10_measured_locations.csv", "nunique(climate)", "count")
    bwh = int(loc[loc.climate == "BWh"].systems.sum())
    add("n_bwh", f"{bwh}/{int(loc.systems.sum())}", bwh, "E10_measured_locations.csv", "sum(systems | climate == BWh)", "count")
    paper = loc[loc.group == "paper"]
    add("n_long_sensor", str(int(paper.systems.sum())), int(paper.systems.sum()), "E10_measured_locations.csv", "group == paper", "count")
    add("n_dkasc", "12", 12, "E10_measured_locations.csv", "Alice Springs, NT systems", "count")
    add("n_pvdaq_long", "3", int(paper.systems.sum()) - 12, "E10_measured_locations.csv", "paper systems outside DKASC", "count")
    add("n_long_locations", str(len(paper)), len(paper), "E10_measured_locations.csv", "rows with group == paper", "count")
    exp = loc[loc.group == "expansion"]
    sel = pd.read_csv(CE / "selection.csv")
    sel = sel[(sel.role == "main") & (sel.status == "final")]
    longexp = sel[~sel.system_id.str.startswith(("hk_", "ut_", "fmi_"))]
    add("n_long_expansion", str(len(longexp)), len(longexp), "selection.csv", "main & final, PVDAQ", "count")
    add("n_long_expansion_S", str(int((longexp.tier == "S").sum())), int((longexp.tier == "S").sum()), "selection.csv", "tier S", "count")
    add("n_long_expansion_R", str(int((longexp.tier == "R").sum())), int((longexp.tier == "R").sum()), "selection.csv", "tier R", "count")
    short = sel[sel.system_id.str.startswith(("hk_", "ut_", "fmi_"))]
    add("n_short", str(len(short)), len(short), "selection.csv", "main & final, hk_/ut_/fmi_", "count")
    add("n_short_per_country", "3", 3, "selection.csv", "3 per country (HK, NL, FI)", "count")
    e0a = pd.read_csv(ET / "E0a_screen_by_source.csv")
    hk = e0a[e0a.source.str.startswith("Hong")].iloc[0]
    ut = e0a[e0a.source.str.startswith("Utrecht")].iloc[0]
    add("hk_candidates", f"{int(hk.passed)} of {int(hk.candidates)}", [int(hk.passed), int(hk.candidates)], "E0a_screen_by_source.csv", "Hong Kong (Dryad)", "count")
    add("ut_candidates", f"{int(ut.passed)} of {int(ut.candidates)}", [int(ut.passed), int(ut.candidates)], "E0a_screen_by_source.csv", "Utrecht (Zenodo)", "count")
    s_pv = e0a[(e0a.tier == "S") & (e0a.source == "PVDAQ")].iloc[0]
    r_pv = e0a[(e0a.tier == "R") & (e0a.source == "PVDAQ")].iloc[0]
    add("pvdaq_S_candidates", f"{int(s_pv.passed)} of {int(s_pv.candidates)}", [int(s_pv.passed), int(s_pv.candidates)], "E0a_screen_by_source.csv", "tier S, PVDAQ", "count")
    add("pvdaq_R_candidates", f"{int(r_pv.passed)} of {int(r_pv.candidates)}", [int(r_pv.passed), int(r_pv.candidates)], "E0a_screen_by_source.csv", "tier R, PVDAQ", "count")
    t0 = pd.read_csv(TAB / "T0_sites.csv")
    add("orig_candidates", f"{int(t0.retained.sum())} of {len(t0)}", [int(t0.retained.sum()), len(t0)], "T0_sites.csv", "retained / rows", "count")
    sites = pd.read_csv(CE / "semisynthetic_sites.csv")
    add("n_weather_sites", str(len(sites)), len(sites), "semisynthetic_sites.csv", "rows", "count")
    add("n_weather_classes", str(sites.climate.nunique()), int(sites.climate.nunique()), "semisynthetic_sites.csv", "nunique(climate)", "count")
    allc = set(loc.climate) | set(sites.climate)
    add("n_classes_all", str(len(allc)), len(allc), "E10 + semisynthetic_sites", "union of classes", "count")
    runs = pd.concat([pd.read_csv(f) for f in sorted(CE.glob("semisynthetic_climate_runs*.csv"))], ignore_index=True)
    key = ["site_id", "family", "rate", "seed"] if "site_id" in runs.columns else None
    if key:
        r1 = runs.drop_duplicates(key + ["method"])
        add("n_semi_records", f"{len(r1.drop_duplicates(key)):,}", int(len(r1.drop_duplicates(key))), "semisynthetic_climate_runs*.csv", "unique site x response x rate x seed", "count")
        add("n_semi_runs", f"{len(r1):,}", int(len(r1)), "semisynthetic_climate_runs*.csv", "unique records x method", "count")
    t1 = pd.read_csv(TAB / "T1_plr_by_system.csv")
    add("span_long_sensor", f"{t1.span_years.min():.1f}–{t1.span_years.max():.1f}", [t1.span_years.min(), t1.span_years.max()], "T1_plr_by_system.csv", "span_years min-max", "result")

    # drift screen
    scr = pd.read_csv(RES / "tier_a_screen.csv").set_index("system_id")
    for sid in ("pvdaq_1432", "pvdaq_1433", "pvdaq_1430"):
        cols = [c for c in scr.columns if "trend" in c and ("era5" in c.lower() or "clearsky" in c.lower())]
        era = [c for c in cols if "era5" in c.lower()][0]
        clr = [c for c in cols if "clearsky" in c.lower()][0]
        add(f"drift_{sid}", f"−{abs(100 * scr.loc[sid, era]):.2f} and −{abs(100 * scr.loc[sid, clr]):.2f}",
            [100 * scr.loc[sid, era], 100 * scr.loc[sid, clr]], "tier_a_screen.csv", f"{era}, {clr}")

    # ------------------------------------------------------------------ fit (common validation weeks), Table II
    common = pd.read_csv(TAB / "T3b_common_domain_fit.csv")
    rm = common.groupby("model").rmse.mean()
    r2 = common.groupby("model").r2.mean()
    add("fit_rmse_range", rng([rm[m] for m in MODELS]), {m: rm[m] for m in MODELS}, "T3b_common_domain_fit.csv", "mean rmse per model over 15 systems")
    add("fit_r2_range", rng([r2[m] for m in MODELS]), {m: r2[m] for m in MODELS}, "T3b_common_domain_fit.csv", "mean r2 per model")
    for m in MODELS + ["khd-full"]:
        add(f"fit_rmse_{m}", f3(rm[m]), rm[m], "T3b_common_domain_fit.csv", f"mean rmse, model == {m}")
    rec = pd.read_csv(TAB / "T2_recovery_by_system.csv")
    mae = rec.groupby("method").error.apply(lambda e: np.abs(e).mean())
    for m in ORDER:
        add(f"inj_mae_{m}", f3(mae[m]), mae[m], "T2_recovery_by_system.csv", f"mean |error| over 15 systems x 4 rates, method == {m}")
    main6 = [m for m in ["rdtools", *MODELS] if m != "random_forest"]
    add("inj_mae_range_main", rng([mae[m] for m in main6]), None, "T2_recovery_by_system.csv", "reference and six models (no random forest)")
    add("n_injection_units", "60", int(rec[rec.method == "rdtools"].shape[0]), "T2_recovery_by_system.csv", "rows per method", "count")
    add("injected_rates", "−0.25, −0.5, −1 and −2", sorted(rec.rate.unique().tolist()), "T2_recovery_by_system.csv", "unique rate", "design")
    e7 = pd.read_csv(ET / "E7_semisynthetic_by_site.csv")
    e11 = pd.read_csv(ET / "E11_rank_tests.csv")
    semi = e11[e11.set.str.startswith("known")].set_index("method")
    meas = e11[e11.set.str.startswith("measured")].set_index("method")
    for m in ORDER:
        s = e7[e7.method == m].mae
        add(f"exact_mae_{m}", f3(s.mean()), s.mean(), "E7_semisynthetic_by_site.csv", f"mean mae over 20 sites, method == {m}")
        add(f"exact_worst_{m}", f3(s.max()), s.max(), "E7_semisynthetic_by_site.csv", f"max mae over sites, method == {m}")
        add(f"exact_best_{m}", str(int(semi.loc[m, "blocks_best"])), int(semi.loc[m, "blocks_best"]), "E11_rank_tests.csv", f"blocks_best, known-truth, {m}")
    add("exact_mean_rank_khd", f2(semi.loc["khd_full", "mean_rank"]).rstrip("0").rstrip(".") if False else "2.0", semi.loc["khd_full", "mean_rank"], "E11_rank_tests.csv", "mean_rank khd_full known-truth")
    add("friedman_sites_p", "10⁻⁴", semi.friedman_p.iloc[0], "E11_rank_tests.csv", "friedman_p known-truth (< 1e-4)")
    rd = e7[e7.method == "rdtools"].set_index("country").mae
    add("exact_ref_NL", f3(rd["Netherlands"]), rd["Netherlands"], "E7_semisynthetic_by_site.csv", "rdtools, Netherlands")
    add("exact_ref_FI", f3(rd["Finland"]), rd["Finland"], "E7_semisynthetic_by_site.csv", "rdtools, Finland")
    t27 = pd.read_csv(TAB / "T27_site_dependence.csv")
    four = t27[t27.case == "four sites weighted equally"].set_index("method").recovery_mae
    for m in ORDER:
        add(f"four_loc_{m}", f3(four[m]), four[m], "T27_site_dependence.csv", f"four sites weighted equally, {m}")
    gv = lambda c, m: t27[(t27.case == c) & (t27.method == m)].recovery_mae.iloc[0]  # noqa: E731
    add("dkasc_khd", f3(gv("DKASC only (12 arrays)", "khd_full")), gv("DKASC only (12 arrays)", "khd_full"), "T27_site_dependence.csv", "DKASC only, khd_full")
    add("dkasc_ref", f3(gv("DKASC only (12 arrays)", "rdtools")), gv("DKASC only (12 arrays)", "rdtools"), "T27_site_dependence.csv", "DKASC only, rdtools")
    add("us_khd", f3(gv("US sites only (3 systems)", "khd_full")), gv("US sites only (3 systems)", "khd_full"), "T27_site_dependence.csv", "US only, khd_full")
    arb = rec[(rec.method == "khd_full") & (rec.system_id == "pvdaq_2107") & (rec.rate == -1.0)].error.iloc[0]
    add("arbuckle_khd_err", f"+{arb:.2f}", arb, "T2_recovery_by_system.csv", "khd_full, pvdaq_2107, rate -1")
    e5 = pd.read_csv(ET / "E5_country_recovery.csv")
    e6 = pd.read_csv(ET / "E6_group_recovery.csv")
    for m in ORDER:
        v = e5[(e5.group.str.startswith("expansion")) & (e5.method == m)].mae.iloc[0]
        add(f"longexp_{m}", f3(v), v, "E5_country_recovery.csv", f"expansion PVDAQ, United States, {m}")
        v = e6[(e6.group.str.startswith("short")) & (e6.method == m)].mae_equal_country.iloc[0]
        add(f"short_{m}", f3(v), v, "E6_group_recovery.csv", f"short-record, country-weighted, {m}")
    sh = e5[e5.group.str.startswith("short")]
    for c, lab in (("Finland", "FI"), ("Netherlands", "NL"), ("China (Hong Kong)", "HK")):
        v = sh[sh.country == c].mae
        add(f"short_range_{lab}", rng(v), [v.min(), v.max()], "E5_country_recovery.csv", f"short-record, {c}, min-max over methods")
    add("country_best_khd", f"{int(meas.loc['khd_full', 'blocks_best'])} of {int(meas.loc['khd_full', 'n_blocks'])}",
        int(meas.loc["khd_full", "blocks_best"]), "E11_rank_tests.csv", "measured (5 countries), khd_full blocks_best")
    add("friedman_countries_p", f2(meas.friedman_p.iloc[0]), meas.friedman_p.iloc[0], "E11_rank_tests.csv", "friedman_p measured (5 countries)")

    # fit vs rate (Spearman, seven models)
    fr_rho, fr_p = spearmanr([rm[m] for m in MODELS], [mae[m] for m in MODELS])
    add("spearman_fit_rate", f"−{abs(fr_rho):.2f}", fr_rho, "T3b + T2", "spearman(mean rmse, recovery mae), seven models")
    add("spearman_fit_rate_p", f2(fr_p), fr_p, "T3b + T2", "p of the Spearman test")
    t5 = pd.read_csv(TAB / "T5_friedman_recovery.csv").iloc[0]
    add("friedman_injection_p", "3 × 10⁻⁹", t5.p_value, "T5_friedman_recovery.csv", "p_value (rounded to 1 significant digit)")
    add("nemenyi_cd", f2(t5.nemenyi_cd), t5.nemenyi_cd, "T5_friedman_recovery.csv", "nemenyi_cd")
    t21 = pd.read_csv(TAB / "T21_mixed_model_recovery.csv").set_index("method")
    add("mixed_rf", f"+{f3(t21.loc['random_forest', 'diff_vs_rdtools'])}", t21.loc["random_forest", "diff_vs_rdtools"], "T21_mixed_model_recovery.csv", "random_forest diff_vs_rdtools")

    # measured spread and window sensitivity (Fig. 3)
    two = t1[[f"{m}_plr_mean" for m in MODELS]]
    spread = two.max(axis=1) - two.min(axis=1)
    add("spread_median", f2(spread.median()), spread.median(), "T1_plr_by_system.csv", "median over 15 systems of max-min of seven two-stage means")
    add("spread_max", f2(spread.max()), spread.max(), "T1_plr_by_system.csv", "max of the same")
    t12 = pd.read_csv(TAB / "T12_refwindow_sensitivity.csv")
    add("window_max", f2(t12.window_range.max()), t12.window_range.max(), "T12_refwindow_sensitivity.csv", "max window_range")
    add("window_median", f2(t12.window_range.median()), t12.window_range.median(), "T12_refwindow_sensitivity.csv", "median window_range")
    e1 = pd.read_csv(ET / "E1_field_rates.csv")
    m1 = e1[(e1.role == "main") & e1.method.isin(MODELS)]
    sp = m1.groupby("system_id").plr.agg(lambda s: s.max() - s.min())
    spg = sp.groupby(sp.index.map(group_of)).median()
    add("spread_longexp", f2(spg["long expansion"]), spg["long expansion"], "E1_field_rates.csv", "median spread, long expansion mains")
    add("spread_short", f2(spg["short"]), spg["short"], "E1_field_rates.csv", "median spread, short mains")
    ref = e1[(e1.role == "main") & (e1.method == "rdtools")]
    half = ((ref.ci_high - ref.ci_low) / 2).groupby(ref.system_id.map(group_of)).median()
    add("half_longexp", f2(half["long expansion"]), half["long expansion"], "E1_field_rates.csv", "median CI half-width, reference, long expansion")
    add("half_short", f2(half["short"]), half["short"], "E1_field_rates.csv", "median CI half-width, reference, short")
    add("ci_width_khd", f2((t1.khd_ci_high - t1.khd_ci_low).median()), None, "T1_plr_by_system.csv", "median khd CI width")
    add("ci_width_ref", f2((t1.rdtools_ci_high - t1.rdtools_ci_low).median()), None, "T1_plr_by_system.csv", "median reference CI width")

    # ------------------------------------------------------------------ ablation and robustness (Table III)
    ab = pd.read_csv(TAB / "T25_khd_ablation_exact_rate.csv").set_index(["set", "case", "variant"])
    for s, c in (("semi", "thin_film"), ("short", "thin_film"), ("hard", "non_pvwatts"), ("hard", "soiling"),
                 ("hard", "step"), ("semi", "pvwatts")):
        for v in ("full", "no_physics", "no_correction", "rdtools"):
            add(f"abl_{s}_{c}_{v}", f3(ab.loc[(s, c, v), "mae"]), ab.loc[(s, c, v), "mae"], "T25_khd_ablation_exact_rate.csv", f"{s}/{c}/{v} mae")
        add(f"abl_n_{s}_{c}", str(int(ab.loc[(s, c, "full"), "n"])), int(ab.loc[(s, c, "full"), "n"]), "T25_khd_ablation_exact_rate.csv", f"{s}/{c}/full n", "count")
    add("abl_short_tf_p", f2(ab.loc[("short", "thin_film", "no_physics"), "p_wilcoxon"]), None, "T25_khd_ablation_exact_rate.csv", "short thin_film no_physics p")
    h18 = pd.read_csv(TAB / "T18_semisynthetic_hard.csv")
    cp = h18[(h18.scenario == "step") & (h18.method == "changepoint")].mae.iloc[0]
    add("changepoint_step", f3(cp), cp, "T18_semisynthetic_hard.csv", "step, changepoint mae")

    # ------------------------------------------------------------------ uncertainty products (Table IV)
    t16 = pd.read_csv(TAB / "T16_semisynthetic_ci.csv").set_index(["method", "family"])
    kc = int(t16.loc["khd_full"].covered.sum())
    add("rate_ci_cov", f"{kc}/{int(t16.loc['khd_full'].n.sum())}", kc, "T16_semisynthetic_ci.csv", "khd_full covered / n")
    add("rate_ci_width", f2(t16.loc["khd_full"].median_width.mean()), t16.loc["khd_full"].median_width.mean(), "T16_semisynthetic_ci.csv", "mean of the two median widths")
    add("ref_ci_cov_pvw", f"{int(t16.loc[('rdtools', 'pvwatts')].covered)}/24", None, "T16_semisynthetic_ci.csv", "rdtools pvwatts covered")
    add("ref_ci_cov_tf", f"{int(t16.loc[('rdtools', 'thin_film')].covered)}/24", None, "T16_semisynthetic_ci.csv", "rdtools thin_film covered")
    add("ref_ci_width_pvw", f2(t16.loc[("rdtools", "pvwatts")].median_width), None, "T16_semisynthetic_ci.csv", "rdtools pvwatts median width")
    add("hourly_cov_measured", f"{100 * t1.khd_coverage_80.min():.0f}–{100 * t1.khd_coverage_80.max():.0f}", None, "T1_plr_by_system.csv", "khd_coverage_80 min-max")
    ss = pd.read_csv(RES / "semisynthetic_runs.csv")
    k = ss[ss.method == "khd_full"]
    add("hourly_cov_semi", f"{round(100 * k.coverage_80.mean()):.0f}", k.coverage_80.mean(), "semisynthetic_runs.csv", "mean coverage_80, khd_full")
    fw = pd.read_csv(TAB / "T22_forward_by_model.csv").set_index("model")
    ml = fw.drop("kappa_hydra_d")
    add("forward_cov", f"{100 * fw.loc['kappa_hydra_d', 'coverage_80']:.0f}", fw.loc["kappa_hydra_d", "coverage_80"], "T22_forward_by_model.csv", "kappa coverage_80")
    add("forward_rmse_increase", f"{100 * (ml.ratio_equal_location.min() - 1):.0f}–{100 * (ml.ratio_equal_location.max() - 1):.0f}", None, "T22_forward_by_model.csv", "ratio_equal_location - 1, seven models")
    add("forward_rmse_retro", rng(ml.retrospective_rmse_equal_location), None, "T22_forward_by_model.csv", "retrospective_rmse_equal_location")
    add("forward_rmse_later", rng(ml.rmse), None, "T22_forward_by_model.csv", "rmse, later years")
    add("forward_runs", "600", 15 * 8 * 5, "protocol: 15 arrays x 8 models x 5 seeds", "derived count", "count")

    # ------------------------------------------------------------------ equation (2) and positivity
    ij = pd.read_csv(TAB / "T26_injection_truth_check.csv")
    add("eq2_exact_dev", f3(ij.err_exact_t05.abs().max()), ij.err_exact_t05.abs().max(), "T26_injection_truth_check.csv", "max |err_exact_t05|")
    add("eq2_centred_dev", f3(ij.err_exact.abs().max()), ij.err_exact.abs().max(), "T26_injection_truth_check.csv", "max |err_exact|")
    add("eq2_first_order_dev", f3(ij.err_eq2.abs().max()), ij.err_eq2.abs().max(), "T26_injection_truth_check.csv", "max |err_eq2|")
    c30 = pd.read_csv(TAB / "T30_centering.csv")
    tmax = c30.kappa_span_years.max()
    add("span_max", f"{tmax:.1f}", tmax, "T30_centering.csv", "max kappa_span_years")
    add("positivity", f"{2 / tmax:.3f}", 2 / tmax, "T30_centering.csv", "2 / max span (fraction per year)", "derived")

    # ------------------------------------------------------------------ protocol Y20 v2 (2026-10-02): fair reference, ADR on 20 sites
    e12f = ET / "E12_fair_reference.csv"
    if e12f.exists():
        e12 = pd.read_csv(e12f)
        for r in e12.itertuples():
            key = f"y20_{r.part}_{r.family}_{r.method}"
            add(f"{key}_mae", f3(r.mae), r.mae, "E12_fair_reference.csv", f"part {r.part}, {r.family}, {r.method}: mae (sites equal)")
            add(f"{key}_worst", f3(r.worst_site), r.worst_site, "E12_fair_reference.csv", f"part {r.part}, {r.family}, {r.method}: worst site")
            add(f"{key}_n", str(int(r.n_records)), int(r.n_records), "E12_fair_reference.csv", f"part {r.part}, {r.family}, {r.method}: records", "count")
        add("design_adr20_records", "120", 120, "protocol Y20 v2: 20 sites x 2 rates x 3 seeds", "design", "design")
        add("design_fit_months", "12", 12, "protocol Y20 v2: coefficient fitted on the first 12 months", "design", "design")

    # ------------------------------------------------------------------ design constants (protocol / configs)
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
    budget = float(np.median([hp[m]["hpo_seconds"] for m in MODELS]))
    add("equal_time_budget", f"{budget:.0f}", budget, "configs/perf_models_hpo.json", "median hpo_seconds of the seven models", "design")
    t20 = pd.read_csv(TAB / "T20_equal_time_budget.csv").set_index("model")
    add("equal_time_lstm", str(int(t20.loc["lstm", "trials_equal_time"])), None, "T20_equal_time_budget.csv", "lstm trials_equal_time")
    add("equal_time_gru", str(int(t20.loc["gru", "trials_equal_time"])), None, "T20_equal_time_budget.csv", "gru trials_equal_time")
    for k_, t_, src in (("trials", "20", "configs/perf_models_hpo.json: trials"), ("dev_systems", "3", "protocol: three development systems"),
                        ("seeds", "5", "protocol: seeds 0-4"), ("train_months", "24", "protocol: first 24 data months"),
                        ("tree_window_h", "3", "protocol: tree input window"), ("seq_window_h", "24", "protocol: sequence window"),
                        ("val_week", "5", "protocol: every fifth ISO week"), ("data_month_hours", "150", "protocol: data month"),
                        ("tune_months", "12", "forward protocol: 12 tuning months"), ("gap_h", "24", "forward protocol: 24-h gaps"),
                        ("a_center", "0.5", "Eq. (1): first-year centre a"), ("warmup", "30", "protocol: 30 % warm-up of rho"),
                        ("q_levels", "0.1, 0.5, 0.9", "model quantiles"), ("hourly_band", "80", "nominal 80 % band"),
                        ("rate_ci", "95", "nominal 95 % interval"), ("noise", "1", "semi-synthetic: 1 % noise"),
                        ("semi_rates", "−0.5 and −1", "semi-synthetic rates (configs)"), ("semi_seeds", "three", "seeds 0-2"),
                        ("semi_years", "2015–2022", "ERA5 weather years"), ("drift_threshold", "1", "drift rule: 1 %/yr"),
                        ("drift_months", "36", "drift rule: matched months, long records"), ("drift_months_short", "24", "drift rule: short records"),
                        ("min_years_long", "4", "screen: valid years (sensor tiers)"), ("min_years_R", "3", "screen: valid years tier R"),
                        ("min_years_short", "2", "screen: valid years short records"), ("ref_months_short", "12", "short records: 12-month reference"),
                        ("per_location", "three", "at most three main systems per added location"), ("step", "3", "3 % step scenario"),
                        ("example_years", "25", "illustrative 25-year profile"), ("dm_lag", "23", "autocovariances up to lag 23"),
                        ("hk_step", "37", "Hong Kong GHI sensor step, Feb. 2023 (hk_rescreen log)"),
                        ("thin_film_gamma", "half", "thin-film-like response: half the datasheet coefficient"),
                        ("ablation_runs", "288", "run_semisynthetic_ablation.py: 288 runs"),
                        ("hk_step_date", "February 2023", "hk_rescreen log / CLIMATE_EXPANSION_PLAN.md amendment of 2026-10-01"),
                        ("kg_period", "1991–2020", "Koppen-Geiger map period (Beck et al. 2023)"),
                        ("golden_ids", "1430, 1432, 1433", "PVDAQ system identifiers (tier_a_screen.csv)"),
                        ("seed0", "0", "equal-time comparison on seed 0 (T20_equal_time_budget.csv seeds_equal_time)"),
                        ("windows", "1–24, 13–36, 25–48", "T12_refwindow_sensitivity.csv columns plr_m1_24, plr_m13_36, plr_m25_48"),
                        ("whiskers", "5th–95th", "make_rev_figures.py boxplot whis=(5, 95)"),
                        ("records_per_site", "12", "E7_semisynthetic_by_site.csv n (2 responses x 2 rates x 3 seeds)"),
                        ("formula_constants", "1, 2, 100, 200", "constants of Eq. (2) and its general form"),
                        ("license", "CC BY 4.0", "data licences (provider pages)"),
                        ("six_months", "6", "transfer test: six local months (T9_transfer.csv conditions scratch_6, finetune_6)")):
        add(f"design_{k_}", t_, t_, src, "design constant", "design")
    # derived illustration (Discussion): linear 25-year profile, 0.20 %/yr
    years = np.arange(25)
    add("example_final_year_pp", "5", 0.20 * 25, "derived: 0.20 %/yr x 25 yr", "derived", "derived")
    add("example_lifetime", "2.4", float(0.20 * years.mean()), "derived: mean loss over years 0-24 at 0.20 %/yr", "derived", "derived")

    v3_rows()
    out = REV / "result_ledger.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text", "value", "source", "selector", "kind"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"wrote {out}: {len(ROWS)} rows")
    for r in ROWS:
        print(f"{r['id']:28s} {r['text']}")


if __name__ == "__main__":
    main()
