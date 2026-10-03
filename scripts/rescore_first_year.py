"""Re-score every known-rate experiment against the generator truth at the first-year level (t = 0.5 yr), the
convention in which all estimators report (src/degradation/truth_convention.py). No model is refitted; only the
truth changes. Outputs results/first_year/FY_*.csv:

  FY_semi_4site        4 weather sites, PVWatts / thin-film-like, by method (bias, MAE, n)
  FY_coverage_4site    95 % rate-interval coverage on the 48 records (Wilson interval), width, interval score
  FY_ablation          kappa-HYDRA-D variants and the reference on complete, assumption-violating and 3-year records
  FY_hard              assumption-violating records, all methods (bias, MAE)
  FY_drift             identifiability test with and without weather drift
  FY_climate_site      20 weather sites x method (site MAE)
  FY_fair              reference with fitted / true coefficient and ADR records (20 sites)
  FY_shift             size of the convention change per truth value
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation.truth_convention import SCENARIO_M05, first_year  # noqa: E402

DP = ROOT / "results/degradation_paper"
CL = ROOT / "results/climate_expansion"
OUT = ROOT / "results/first_year"


def read(base: Path, pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted(base.glob(pattern))], ignore_index=True)


def summ(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)
    return pd.DataFrame({"n": g.size(), "bias": g.err.mean(), "mae": g.err.apply(lambda e: e.abs().mean())}).reset_index()


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = c / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"truth_start": [-0.25, -0.5, -1.0, -2.0]}).assign(
        truth_first_year=lambda d: first_year(d.truth_start), shift=lambda d: first_year(d.truth_start) - d.truth_start
    ).to_csv(OUT / "FY_shift.csv", index=False)

    # 4 weather sites (semisynthetic_runs + kappa CI runs)
    s = read(DP, "semisynthetic_runs.csv").drop_duplicates(["method", "weather", "family", "rate", "seed"], keep="last")
    s["truth"] = first_year(s.rate)
    s["err"] = s.plr - s.truth
    summ(s, ["method", "family"]).to_csv(OUT / "FY_semi_4site.csv", index=False)
    ci = read(DP, "semisynthetic_ci_runs_*.csv").drop_duplicates(["weather", "family", "rate", "noise_seed"], keep="last")
    ci = ci.rename(columns={"noise_seed": "seed"}).assign(method="khd_full")
    cov_src = pd.concat([ci[["method", "weather", "family", "rate", "seed", "plr", "ci_low", "ci_high"]],
                         s[s.method != "khd_full"][["method", "weather", "family", "rate", "seed", "plr", "ci_low", "ci_high"]]])
    cov_src["truth"] = first_year(cov_src.rate)
    rows = []
    for (m, fam), d in cov_src.dropna(subset=["ci_low"]).groupby(["method", "family"]):
        covered = int(((d.truth >= d.ci_low) & (d.truth <= d.ci_high)).sum())
        lo, hi = wilson(covered, len(d))
        width = d.ci_high - d.ci_low
        miss = np.maximum(d.ci_low - d.truth, 0) + np.maximum(d.truth - d.ci_high, 0)
        rows.append({"method": m, "family": fam, "n": len(d), "covered": covered, "coverage": covered / len(d),
                     "wilson_low": lo, "wilson_high": hi, "median_width": width.median(), "mean_width": width.mean(),
                     "interval_score": (width + 2 / 0.05 * miss).mean(), "mae": (d.plr - d.truth).abs().mean()})
    pd.DataFrame(rows).to_csv(OUT / "FY_coverage_4site.csv", index=False)

    # assumption-violating records, all methods
    h = read(DP, "semisynthetic_hard_runs_*.csv").drop_duplicates(["weather", "scenario", "seed", "method"], keep="last")
    h["truth"] = first_year(h.truth_avg, h.scenario.map(SCENARIO_M05))
    h["err"] = h.plr - h.truth
    summ(h, ["scenario", "method"]).to_csv(OUT / "FY_hard.csv", index=False)

    # ablation: full model and reference from the main runs, variants from the ablation runs
    a = read(DP, "semisynthetic_ablation_runs.csv").drop_duplicates(["set", "weather", "case", "rate", "seed", "variant"], keep="last")
    hard_rows = a.set == "hard"
    a.loc[hard_rows, "truth_fy"] = first_year(a.loc[hard_rows, "truth"], a.loc[hard_rows, "case"].map(SCENARIO_M05))
    a.loc[~hard_rows, "truth_fy"] = first_year(a.loc[~hard_rows, "truth"])
    a["err"] = a.plr - a.truth_fy
    full_semi = s[s.method.isin(["khd_full", "rdtools"])].assign(
        set="semi", case=lambda d: d.family, variant=lambda d: d.method.map({"khd_full": "full", "rdtools": "rdtools"}))
    full_hard = h[h.method.isin(["khd_full", "rdtools"])].assign(
        set="hard", case=lambda d: d.scenario, variant=lambda d: d.method.map({"khd_full": "full", "rdtools": "rdtools"}), rate=-99.0)
    cols = ["set", "weather", "case", "rate", "seed", "variant", "err"]
    ab = pd.concat([a[cols], full_semi[cols], full_hard[cols]], ignore_index=True)
    ab["key"] = ab.weather + "|" + ab.case + "|" + ab.rate.astype(str) + "|" + ab.seed.astype(str)
    rows = []
    for (st, case), d in ab.groupby(["set", "case"]):
        full = d[d.variant == "full"].set_index("key").err
        for v, dv in d.groupby("variant"):
            e = dv.set_index("key").err
            row = {"set": st, "case": case, "variant": v, "n": len(e), "bias": e.mean(), "mae": e.abs().mean()}
            if v != "full":
                k = e.index.intersection(full.index)
                row["mae_minus_full"] = e[k].abs().mean() - full[k].abs().mean()
                diff = e[k].abs() - full[k].abs()
                row["p_wilcoxon"] = float(wilcoxon(diff).pvalue) if len(k) > 5 and (diff != 0).any() else np.nan
                row["n_paired"] = len(k)
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "FY_ablation.csv", index=False)

    # identifiability with weather drift
    i = read(DP, "identifiability_runs.csv").drop_duplicates(["weather", "family", "rate", "seed", "drift", "method"], keep="last")
    i["err"] = i.plr - first_year(i.rate)
    summ(i, ["drift", "family", "method"]).to_csv(OUT / "FY_drift.csv", index=False)

    # 20 weather sites
    c = pd.concat([read(CL, "semisynthetic_climate_runs*.csv"), read(CL, "info_equal_runs*.csv")]).dropna(subset=["plr"])
    c = c.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    c["err"] = c.plr - first_year(c.rate)
    c.groupby(["site_id", "method"]).err.apply(lambda e: e.abs().mean()).rename("site_mae").reset_index().to_csv(
        OUT / "FY_climate_site.csv", index=False)
    f = read(CL, "fair_reference_runs*.csv").dropna(subset=["plr"]).drop_duplicates(
        ["part", "site_id", "family", "rate", "seed", "method"], keep="last")
    f["err"] = f.plr - first_year(f.truth)
    rows = []
    for (part, fam, m), d in pd.concat([f, f[f.part == "A"].assign(family="both")]).groupby(["part", "family", "method"]):
        site = d.groupby("site_id").err.apply(lambda e: e.abs().mean())
        rows.append({"part": part, "family": fam, "method": m, "n_records": len(d), "mae": site.mean(), "bias": d.err.mean(),
                     "worst_site": site.max(), "worst_site_id": site.idxmax()})
    pd.DataFrame(rows).to_csv(OUT / "FY_fair.csv", index=False)
    for name in ("FY_semi_4site", "FY_coverage_4site", "FY_ablation", "FY_hard", "FY_fair"):
        print("\n" + name)
        print(pd.read_csv(OUT / f"{name}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
