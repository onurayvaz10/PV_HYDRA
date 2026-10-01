"""Paper tables for the degradation study. Reads only result files; writes
results/degradation_paper/tables/*.csv and a short markdown digest (DIGEST.md).

T1  PLR per system: RdTools [CI], each two-stage model (mean, SD over 5 seeds), kappa-HYDRA-D [CI]
T2  Known-truth recovery: error = dPLR - expected, expected = r + 2 d r t_c / 100 (method's own d, t_c);
    T2_recovery_by_system.csv keeps every system x rate x method
T3  Performance-model fit on each model's held-out reference weeks, plus parameters and HPO time
T3b Common-domain fit: all models on the SAME held-out hours
T4  Diebold-Mariano (HLN, squared error, 24-h lag), Holm-adjusted per system, vs XGBoost and vs kappa-HYDRA-D
T5  Friedman test of recovery error across methods (blocks = system x rate)
T6  kappa-HYDRA-D ablation
T8  Hot/arid vs other climates on Tier B (system performance loss)
T9  Transfer for young plants
T11 Semi-synthetic benchmark (exact truth on real weather); T10 bin-diagnostic note
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.stats import diebold_mariano_hln, friedman, holm

RES = ROOT / "results/degradation_paper"
TAB = RES / "tables"


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)])


def _pm(values: pd.Series, digits: int = 2) -> str:
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}" if len(values) > 1 else f"{values.mean():.{digits}f}"


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(RES / "reference_plr.csv")
    ref = ref[ref.status == "ok"]
    parts = [pd.read_csv(p) for p in sorted(RES.glob("ml_plr_runs*.csv"))]      # one file per parallel worker
    runs = pd.concat(parts, ignore_index=True).drop_duplicates(["system_id", "model", "seed", "injection"]) if parts else pd.DataFrame()
    kd = pd.read_csv(RES / "kappa_hydra_d_runs.csv") if (RES / "kappa_hydra_d_runs.csv").exists() else pd.DataFrame()
    rd_inj = pd.read_csv(RES / "rdtools_injected.csv") if (RES / "rdtools_injected.csv").exists() else pd.DataFrame()
    digest = ["# Degradation study — result digest (generated)", ""]

    # T1 ------------------------------------------------------------------
    t1 = ref[["system_id", "label", "climate", "plr", "ci_low", "ci_high", "span_years"]].rename(
        columns={"plr": "rdtools_plr", "ci_low": "rdtools_ci_low", "ci_high": "rdtools_ci_high"})
    if len(runs):
        base = runs[runs.injection == 0]
        wide = base.groupby(["system_id", "model"]).plr.agg(["mean", "std"]).unstack("model")
        wide.columns = [f"{m}_plr_{s}" for s, m in wide.columns]
        t1 = t1.merge(wide.reset_index(), on="system_id", how="left")
    if len(kd):
        full = kd[(kd.variant == "full") & (kd.injection == 0)][["system_id", "plr", "ci_low", "ci_high", "coverage_80"]]
        t1 = t1.merge(full.rename(columns={"plr": "khd_plr", "ci_low": "khd_ci_low", "ci_high": "khd_ci_high",
                                           "coverage_80": "khd_coverage_80"}), on="system_id", how="left")
    t1.to_csv(TAB / "T1_plr_by_system.csv", index=False)
    methods = [c[:-9] for c in t1.columns if c.endswith("_plr_mean")] + (["khd"] if "khd_plr" in t1 else [])
    digest += ["## Agreement with RdTools (inj. 0, across systems)", "", "| method | mean |Δ| %/yr | Spearman ρ | n |", "|---|---|---|---|"]
    for m in methods:
        col = f"{m}_plr_mean" if m != "khd" else "khd_plr"
        ok = t1[["rdtools_plr", col]].dropna()
        if len(ok) >= 3:
            digest.append(f"| {m} | {np.mean(np.abs(ok[col] - ok.rdtools_plr)):.3f} | "
                          f"{ok[col].corr(ok.rdtools_plr, method='spearman'):.2f} | {len(ok)} |")

    # T2 ------------------------------------------------------------------
    # Expected change for a multiplicative injection under first-year-normalised rates:
    # dPLR = r + 2 d r t_c / 100 (%/yr), d = the method's OWN baseline PLR, t_c = time centre of its data.
    # exact form with the measured first-year centring of each system (src/degradation/injection_truth.py, T30);
    # the former first-order form r + 2 d r tc / 100 omitted the recentring (up to 0.06 %/yr, T26)
    from src.degradation.injection_truth import expected_for

    rows = []

    def add(method, sid, table):
        base = table[table.injection == 0]
        if base.empty:
            return
        d, tc = float(base.plr.iloc[0]), float(base.t_center_years.iloc[0])
        for _, r in table[table.injection != 0].iterrows():
            rows.append({"method": method, "system_id": sid, "rate": r.injection,
                         "error": (r.plr - d) - expected_for(method, sid, r.injection, d, tc),
                         "naive_error": (r.plr - d) - r.injection})

    if len(rd_inj):
        for sid, g in rd_inj.groupby("system_id"):
            add("rdtools", sid, g)
    if len(runs):
        seed_mean = runs.groupby(["system_id", "model", "injection"])[["plr", "t_center_years"]].mean().reset_index()
        for (sid, model), g in seed_mean.groupby(["system_id", "model"]):
            add(model, sid, g)
    if len(kd):
        for (sid, variant), g in kd.groupby(["system_id", "variant"]):
            add(f"khd_{variant}", sid, g)
    rec = pd.DataFrame(rows).dropna()
    if len(rec):
        rec.to_csv(TAB / "T2_recovery_by_system.csv", index=False)
        t2 = rec.groupby(["method", "rate"]).agg(bias=("error", "mean"), mae=("error", lambda e: np.abs(e).mean()),
                                                 rmse=("error", lambda e: np.sqrt((e ** 2).mean())),
                                                 naive_mae=("naive_error", lambda e: np.abs(e).mean()),
                                                 n=("error", "size")).reset_index()
        t2.to_csv(TAB / "T2_synthetic_recovery.csv", index=False)
        overall = rec.groupby("method").error.agg(mae=lambda e: np.abs(e).mean(), bias="mean").sort_values("mae")
        digest += ["", "## Known-truth recovery |ΔPLR − expected| (%/yr), all rates", "", "| method | MAE | bias |", "|---|---|---|"]
        digest += [f"| {m} | {v.mae:.3f} | {v.bias:+.3f} |" for m, v in overall.iterrows()]
        # T5: Friedman on complete blocks, main methods only (ablations excluded)
        main_m = rec[~rec.method.str.startswith("khd_") | (rec.method == "khd_full")]
        block = main_m.assign(block=main_m.system_id + "|" + main_m.rate.astype(str)).pivot_table(
            index="block", columns="method", values="error", aggfunc=lambda e: np.abs(e).mean()).dropna()
        if len(block) >= 3 and block.shape[1] >= 3:
            fr = friedman({m: block[m].tolist() for m in block.columns})
            # Nemenyi post-hoc critical difference of mean ranks (alpha = 0.05): q_alpha = studentized range / sqrt(2)
            from scipy.stats import studentized_range
            k_m, n_b = block.shape[1], len(block)
            fr["nemenyi_cd"] = float(studentized_range.ppf(0.95, k_m, np.inf) / np.sqrt(2) * np.sqrt(k_m * (k_m + 1) / (6 * n_b)))
            ranks = fr.get("mean_rank", {})
            fr["nemenyi_pairs"] = {f"{a} vs {b}": round(ranks[b] - ranks[a], 3) for a in ranks for b in ranks
                                   if ranks[b] - ranks[a] > fr["nemenyi_cd"]}
            pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in fr.items()}]).to_csv(
                TAB / "T5_friedman_recovery.csv", index=False)
            digest += ["", f"Friedman (recovery error, {len(block)} blocks, {block.shape[1]} methods): {fr}"]

    # T3 ------------------------------------------------------------------
    if len(runs):
        base = runs[runs.injection == 0]
        hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())
        t3 = base.groupby("model").agg(val_rmse=("val_rmse", lambda v: _pm(v, 4)), val_mae=("val_mae", lambda v: _pm(v, 4)),
                                       val_r2=("val_r2", lambda v: _pm(v, 3)), parameters=("parameters", "median"),
                                       train_s_shared_load=("train_seconds", "mean")).reset_index()   # parallel workers; T7 = isolated timing
        t3["hpo_seconds"] = t3.model.map(lambda m: hp.get(m, {}).get("hpo_seconds"))
        t3.to_csv(TAB / "T3_fit_and_cost.csv", index=False)
        digest += ["", "## Performance-model fit and cost", "", _md(t3)]

    # T4 ------------------------------------------------------------------
    # Common-domain scoring: every model on the SAME held-out hours = reference-period validation weeks
    # of the two-stage models intersected with kappa-HYDRA-D's held-out hours (RdTools-filtered hours).
    vp = RES / "val_preds"
    if vp.exists():
        def to_ns(a):
            a = np.asarray(a, dtype="int64")
            return a * 1000 if a.size and np.abs(a).max() < 10 ** 17 else a      # microseconds -> ns

        common_rows, dm_rows = [], []
        systems = sorted({p.stem.rsplit("_", 2)[0] for p in vp.glob("*.npz")})
        for sid in systems:
            series = {}
            for p in vp.glob(f"{sid}_*.npz"):
                model = p.stem[len(sid) + 1:].rsplit("_", 1)[0]
                z = np.load(p)
                series.setdefault(model, []).append(pd.DataFrame({"y": z["y"], "pred": z["pred"]},
                                                                 index=to_ns(z["stamps"])))
            if "xgboost" not in series:
                continue
            frames = {m: pd.concat(v).groupby(level=0).mean() for m, v in series.items()}   # seed-averaged
            idx = None
            for f in frames.values():
                idx = f.index if idx is None else idx.intersection(f.index)
            if idx is None or len(idx) < 100:
                continue
            y = frames["xgboost"].loc[idx, "y"].to_numpy()
            loss = {}
            for m, f in frames.items():
                pred = f.loc[idx, "pred"].to_numpy()
                err = pred - y
                loss[m] = err ** 2
                common_rows.append({"system_id": sid, "model": m, "n_common": len(idx), "rmse": float(np.sqrt(np.mean(err ** 2))),
                                    "mae": float(np.mean(np.abs(err))),
                                    "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))})
            for ref_model in ("xgboost", "khd-full"):
                if ref_model not in loss:
                    continue
                pvals, stats = {}, {}
                for m in loss:
                    if m != ref_model:
                        res = diebold_mariano_hln(loss[m], loss[ref_model], 24)
                        pvals[m], stats[m] = res["p_value"], res["dm_hln"]
                adj = holm(pvals)
                dm_rows += [{"system_id": sid, "reference": ref_model, "model": m, "dm_stat": stats[m],
                             "p_holm": adj.get(m)} for m in pvals]
        if common_rows:
            common = pd.DataFrame(common_rows)
            common.to_csv(TAB / "T3b_common_domain_fit.csv", index=False)
            agg = common.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std"]).round(4)
            agg.columns = [f"{a}_{b}" for a, b in agg.columns]
            digest += ["", "## Common-domain fit (same held-out hours for all models)", "", _md(agg.reset_index())]
        if dm_rows:
            dm = pd.DataFrame(dm_rows)
            dm.to_csv(TAB / "T4_dm_tests.csv", index=False)
            summary = dm.assign(sig_better=(dm.dm_stat < 0) & (dm.p_holm < 0.05),
                                sig_worse=(dm.dm_stat > 0) & (dm.p_holm < 0.05)).groupby(["reference", "model"])[
                ["sig_better", "sig_worse"]].sum().reset_index()
            digest += ["", "## DM-HLN (Holm) — systems where model is significantly better / worse than reference", "",
                       _md(summary)]

    # T6 ------------------------------------------------------------------
    if len(kd):
        t6 = kd[kd.injection == 0].groupby("variant").agg(plr=("plr", "mean"), plr_seed_sd=("plr_seed_sd", "mean"),
                                                          val_rmse=("val_rmse", "mean"), coverage_80=("coverage_80", "mean")).reset_index()
        t6.to_csv(TAB / "T6_khd_ablation.csv", index=False)
        digest += ["", "## κ-HYDRA-D ablation (inj. 0, mean over systems)", "", _md(t6)]

    # T9 ------------------------------------------------------------------
    # Transfer for young plants: agreement with RdTools (measured record) and recovery of r = -1 %/yr.
    tr_file = RES / "transfer_runs.csv"
    if tr_file.exists():
        tr = pd.read_csv(tr_file)
        sm = tr.groupby(["system_id", "model", "condition", "injection"])[["plr", "t_center_years", "val_rmse"]].mean().reset_index()
        rd_ref = ref.set_index("system_id").plr
        out = []
        for (sid, model, cond), g in sm.groupby(["system_id", "model", "condition"]):
            b, i = g[g.injection == 0], g[g.injection == -1.0]
            if b.empty or i.empty:
                continue
            d, tc = float(b.plr.iloc[0]), float(b.t_center_years.iloc[0])
            exp = expected_for(model, sid, -1.0, d, tc)                               # exact form, see T26/T30
            out.append({"system_id": sid, "model": model, "condition": cond, "val_rmse": float(b.val_rmse.iloc[0]),
                        "abs_diff_rdtools": abs(d - rd_ref.get(sid, np.nan)),
                        "recovery_error": float(i.plr.iloc[0]) - d - exp})
        if out:
            t9 = pd.DataFrame(out)
            t9.to_csv(TAB / "T9_transfer_by_system.csv", index=False)
            agg = t9.groupby(["model", "condition"]).agg(val_rmse=("val_rmse", "mean"),
                                                        mae_vs_rdtools=("abs_diff_rdtools", "mean"),
                                                        recovery_mae=("recovery_error", lambda e: np.abs(e).mean()),
                                                        n=("system_id", "size")).reset_index()
            agg.to_csv(TAB / "T9_transfer.csv", index=False)
            digest += ["", "## Transfer for young plants (r = -1 %/yr recovery; agreement with RdTools)", "", _md(agg)]

    # T11 -----------------------------------------------------------------
    # Semi-synthetic benchmark: exact truth on real weather (bias of each method by response family).
    ss_file = RES / "semisynthetic_runs.csv"
    if ss_file.exists():
        ss = pd.read_csv(ss_file)
        t11 = ss.groupby(["family", "method"]).agg(bias=("error", "mean"), mae=("error", lambda e: np.abs(e).mean()),
                                                   max_abs=("error", lambda e: np.abs(e).max()), n=("error", "size")).reset_index()
        t11.to_csv(TAB / "T11_semisynthetic.csv", index=False)
        digest += ["", "## Semi-synthetic known truth (real weather, exact r): error = PLR - r (%/yr)", "", _md(t11)]
    t10 = TAB / "T10_bin_diagnostic.csv"
    if t10.exists():
        b = pd.read_csv(t10)
        digest += ["", f"## Bin diagnostic: pooled RdTools PLR outside its own POA/T-bin range on "
                       f"{int(b.pooled_outside_bins.sum())} of {len(b)} systems"]

    # T8 ------------------------------------------------------------------
    # Hot/arid vs other climates on Tier B (daily energy, ERA5-normalised): "system performance loss",
    # not module degradation. Reported for all systems and for the plausibility-screened subset.
    tier_b = ROOT / "results/degradation/system_rates.csv"
    if tier_b.exists():
        from scipy import stats as st
        tb = pd.read_csv(tier_b)
        tb["group"] = np.where(tb.climate_class.isin(["BWh", "BWk", "BSh", "BSk"]), "hot/arid", "other")
        out = []
        for subset, sub in (("all", tb), ("screened", tb[tb.status == "applied"])):
            a = 100 * sub[sub.group == "hot/arid"].annual_rate.dropna()
            b = 100 * sub[sub.group == "other"].annual_rate.dropna()
            out.append({"subset": subset, "n_hot_arid": len(a), "median_hot_arid": a.median(), "n_other": len(b),
                        "median_other": b.median(), "mann_whitney_p": st.mannwhitneyu(a, b).pvalue})
        t8 = pd.DataFrame(out)
        t8.to_csv(TAB / "T8_climate_tier_b.csv", index=False)
        digest += ["", "## Hot/arid vs other (Tier B, system performance loss, %/yr)", "", _md(t8)]

    (RES / "DIGEST.md").write_text("\n".join(digest) + "\n", encoding="utf-8")
    print("\n".join(digest))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")        # redirected stdout on Windows defaults to cp1254
    main()
