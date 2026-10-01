"""Tables for the reviewer-driven additions.

T18  harder semi-synthetic scenarios (run_semisynthetic_hard.py): bias and MAE against the average true rate
T19  baselines on measured records (run_measured_baselines.py): agreement with the reference (STL, change-point,
     PVUSA), PVUSA known-truth recovery, and the kappa-HYDRA-D mean-loss diagnostic of the DKASC gap
T20  equal-time tuning budget (run_equal_time.py): trials, validation RMSE and r = -1 %/yr recovery error
T21  linear mixed-effects model of the absolute recovery error (system as random intercept), replacing the
     independence assumption of the Friedman blocks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/degradation_paper"
TAB = RES / "tables"
MODELS = ["xgboost", "random_forest", "lstm", "gru", "tcn", "patchtst", "informer"]


sys.path.insert(0, str(ROOT))
from src.degradation.injection_truth import expected_for  # noqa: E402  exact form, measured centring (T26, T30)


def recovery_rows(table: pd.DataFrame, method: str) -> list[dict]:
    rows = []
    for sid, g in table.groupby("system_id"):
        base = g[g.injection == 0]
        if base.empty:
            continue
        d, tc = float(base.plr.iloc[0]), float(base.t_center_years.iloc[0])
        for _, r in g[g.injection != 0].iterrows():
            rows.append({"method": method, "system_id": sid, "rate": r.injection,
                         "error": (r.plr - d) - expected_for(method, sid, r.injection, d, tc)})
    return rows


def t18() -> None:
    parts = [pd.read_csv(p) for p in sorted(RES.glob("semisynthetic_hard_runs*.csv"))]
    if not parts:
        return
    h = pd.concat(parts, ignore_index=True).drop_duplicates(["weather", "scenario", "seed", "method"], keep="last")
    out = h.groupby(["scenario", "method"]).agg(n=("error", "size"), bias=("error", "mean"),
                                                mae=("error", lambda e: np.abs(e).mean())).reset_index()
    out.to_csv(TAB / "T18_semisynthetic_hard.csv", index=False)
    print("T18\n", out.pivot(index="method", columns="scenario", values="mae").round(3).to_string())


def t19() -> None:
    f = RES / "measured_baselines_runs.csv"
    if not f.exists():
        return
    b = pd.read_csv(f)
    ref = pd.read_csv(RES / "reference_plr.csv").set_index("system_id").plr
    base = b[b.injection == 0].copy()
    base["ref"] = base.system_id.map(ref)
    base["diff"] = base.plr - base.ref
    agree = base.groupby("method").agg(n=("diff", "size"), mean_abs_diff=("diff", lambda x: np.abs(x).mean()),
                                       median_diff=("diff", "median")).reset_index()
    rec = pd.DataFrame(recovery_rows(b[b.method == "pvusa"], "pvusa"))
    kd = pd.read_csv(RES / "kappa_hydra_d_runs.csv")
    full = kd[(kd.variant == "full") & (kd.injection == 0)].set_index("system_id").plr
    ml = base[base.method == "khd_mean_loss"].set_index("system_id").plr
    dk = [s for s in ml.index if s.startswith("dkasc")]
    gap = pd.DataFrame({"system_id": dk, "rdtools": ref[dk].to_numpy(), "khd_full": full[dk].to_numpy(),
                        "khd_mean_loss": ml[dk].to_numpy()})
    gap["gap_full"] = gap.khd_full - gap.rdtools
    gap["gap_mean_loss"] = gap.khd_mean_loss - gap.rdtools
    gap.to_csv(TAB / "T19b_khd_mean_loss_gap.csv", index=False)
    summary = agree.assign(recovery_mae=np.nan)
    if len(rec):
        summary.loc[summary.method == "pvusa", "recovery_mae"] = np.abs(rec.error).mean()
        rec.to_csv(TAB / "T19c_pvusa_recovery_by_system.csv", index=False)
    summary.to_csv(TAB / "T19_measured_baselines.csv", index=False)
    print("T19\n", summary.round(3).to_string(index=False))
    if len(gap):
        print("DKASC gap to RdTools: full median %.3f (less steep %d/%d); mean-loss median %.3f (less steep %d/%d)" % (
            gap.gap_full.median(), (gap.gap_full > 0).sum(), len(gap), gap.gap_mean_loss.median(),
            (gap.gap_mean_loss > 0).sum(), len(gap)))


def t20() -> None:
    eq_f, rates_f = ROOT / "configs/perf_models_hpo_equal_time.json", RES / "equal_time_rates.csv"
    if not eq_f.exists():
        return
    hp, eq = json.loads((ROOT / "configs/perf_models_hpo.json").read_text()), json.loads(eq_f.read_text())
    all_runs = pd.concat([pd.read_csv(p) for p in RES.glob("ml_plr_runs*.csv")], ignore_index=True)
    all_runs = all_runs.drop_duplicates(["system_id", "model", "seed", "injection"], keep="last")
    runs = all_runs[all_runs.seed == 0]
    sf = RES / "equal_time_rates_seeds.csv"
    seeds_t = (pd.read_csv(sf).drop_duplicates(["system_id", "model", "injection", "seed"], keep="last") if sf.exists()
               else pd.DataFrame(columns=["system_id", "model", "injection", "seed", "plr", "t_center_years"]))
    rows = []
    rates = pd.read_csv(rates_f) if rates_f.exists() else pd.DataFrame(columns=["system_id", "model", "injection"])
    rates = rates.drop_duplicates(["system_id", "model", "injection"], keep="last")   # parallel workers may overlap
    for m in MODELS:
        if m not in eq:
            continue
        changed = eq[m]["params"] != hp[m]["params"]
        row = {"model": m, "trials_equal_trial": hp[m]["trials"], "trials_equal_time": eq[m]["trials"],
               "val_rmse_equal_trial": hp[m]["val_rmse"], "val_rmse_equal_time": eq[m]["val_rmse"],
               "config_changed": changed}
        orig = runs[runs.model == m].rename(columns={"injection": "injection"})
        e_orig = recovery_rows(orig[orig.injection.isin([0.0, -1.0])], m)
        row["recovery_mae_r1_equal_trial_seed0"] = np.abs(pd.DataFrame(e_orig).error).mean() if e_orig else np.nan
        if changed:
            e_new = recovery_rows(rates[rates.model == m], m)
            row["recovery_mae_r1_equal_time_seed0"] = np.abs(pd.DataFrame(e_new).error).mean() if e_new else np.nan
        else:
            row["recovery_mae_r1_equal_time_seed0"] = row["recovery_mae_r1_equal_trial_seed0"]
        # five seeds (run_equal_time.py rates --seeds=1,2,3,4): mean over seeds of the per-seed MAE at r = -1 %/yr
        per_seed_trial, per_seed_time = [], []
        for sd in range(5):
            o = all_runs[(all_runs.model == m) & (all_runs.seed == sd) & all_runs.injection.isin([0.0, -1.0])]
            e = recovery_rows(o, m)
            per_seed_trial.append(np.abs(pd.DataFrame(e).error).mean() if e else np.nan)
            if changed:
                n = rates[rates.model == m] if sd == 0 else seeds_t[(seeds_t.model == m) & (seeds_t.seed == sd)]
                complete = n.groupby("system_id").injection.nunique().eq(2).sum() == 15
                e = recovery_rows(n, m) if complete else []
                per_seed_time.append(np.abs(pd.DataFrame(e).error).mean() if e else np.nan)
            else:
                per_seed_time.append(per_seed_trial[-1])
        row["recovery_mae_r1_equal_trial_5seeds"] = float(np.nanmean(per_seed_trial))
        row["recovery_mae_r1_equal_time_5seeds"] = float(np.nanmean(per_seed_time)) if np.isfinite(per_seed_time).any() else np.nan
        row["seeds_equal_time"] = int(np.isfinite(per_seed_time).sum())
        row["recovery_mae_r1_equal_time_seed_sd"] = float(np.nanstd(per_seed_time, ddof=1)) if row["seeds_equal_time"] > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T20_equal_time_budget.csv", index=False)
    print("T20\n", out.round(4).to_string(index=False))


def t25() -> None:
    """κ-HYDRA-D ablation on records with an exact rate (run_semisynthetic_ablation.py), paired with the full model
    and the reference on the same records (semisynthetic_runs.csv, semisynthetic_hard_runs*.csv)."""
    f = RES / "semisynthetic_ablation_runs.csv"
    if not f.exists():
        return
    a = pd.read_csv(f).drop_duplicates(["set", "weather", "case", "rate", "seed", "variant"], keep="last")
    semi = pd.read_csv(RES / "semisynthetic_runs.csv")
    semi = semi[semi.method.isin(["khd_full", "rdtools"])].assign(
        set="semi", case=semi.family, variant=semi.method.map({"khd_full": "full", "rdtools": "rdtools"}))
    hard = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("semisynthetic_hard_runs*.csv"))], ignore_index=True)
    hard = hard.drop_duplicates(["weather", "scenario", "seed", "method"], keep="last")
    hard = hard[hard.method.isin(["khd_full", "rdtools"])].assign(
        set="hard", case=hard.scenario, rate=-99.0, variant=hard.method.map({"khd_full": "full", "rdtools": "rdtools"}))
    cols = ["set", "weather", "case", "rate", "seed", "variant", "error"]
    allr = pd.concat([a[cols], semi[cols], hard[cols]], ignore_index=True)
    out = allr.groupby(["set", "case", "variant"]).agg(n=("error", "size"), bias=("error", "mean"),
                                                         mae=("error", lambda e: np.abs(e).mean())).reset_index()
    # paired test of each ablation against the full model on the same records (Wilcoxon on absolute errors)
    from scipy.stats import wilcoxon
    key = ["set", "weather", "case", "rate", "seed"]
    wide = allr.pivot_table(index=key, columns="variant", values="error").abs()
    p = []
    for (st, cs), g in wide.groupby(level=["set", "case"]):
        for v in ("no_physics", "no_correction", "rdtools"):
            if v in g and "full" in g:
                d = g[["full", v]].dropna()
                diff = d[v] - d["full"]
                pv = wilcoxon(d[v], d["full"]).pvalue if len(d) >= 6 and (diff != 0).any() else np.nan
                p.append({"set": st, "case": cs, "variant": v, "n_paired": len(d),
                          "mae_minus_full": float(diff.mean()), "p_wilcoxon": pv})
    out = out.merge(pd.DataFrame(p), on=["set", "case", "variant"], how="left")
    out.to_csv(TAB / "T25_khd_ablation_exact_rate.csv", index=False)
    print("T25\n", out.round(4).to_string(index=False))


def t21() -> None:
    import statsmodels.formula.api as smf
    rec = pd.read_csv(TAB / "T2_recovery_by_system.csv")
    keep = ["rdtools", *MODELS, "khd_full"]
    rec = rec[rec.method.isin(keep)].copy()
    b = RES / "measured_baselines_runs.csv"
    if b.exists():
        pv = pd.DataFrame(recovery_rows(pd.read_csv(b).query("method == 'pvusa'"), "pvusa"))
        if len(pv):
            rec = pd.concat([rec, pv], ignore_index=True)
    rec["abs_error"] = rec.error.abs()
    model = smf.mixedlm("abs_error ~ C(method, Treatment('rdtools')) + C(rate)", rec, groups=rec["system_id"])
    fit = model.fit(reml=True)
    ci = fit.conf_int()
    rows = []
    for name in fit.params.index:
        if name.startswith("C(method"):
            m = name.split("[T.")[1].rstrip("]")
            rows.append({"method": m, "diff_vs_rdtools": fit.params[name], "ci_low": ci.loc[name, 0],
                         "ci_high": ci.loc[name, 1], "p_value": fit.pvalues[name]})
    out = pd.DataFrame(rows).sort_values("diff_vs_rdtools")
    out["system_variance"] = float(fit.cov_re.iloc[0, 0])
    out["residual_variance"] = float(fit.scale)
    out.to_csv(TAB / "T21_mixed_model_recovery.csv", index=False)
    print("T21 (abs recovery error vs RdTools, random intercept per system)\n", out.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (t18, t19, t20, t21, t25):
        f()
