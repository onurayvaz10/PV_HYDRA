"""Climate expansion, step 4: tables reported by tier (reports/CLIMATE_EXPANSION_PLAN.md).

Tier S and Tier R are summarised separately; a pooled statistic appears only in E4, with a tier term. Locations are
the independent units: errors are averaged within a location first, then over locations (equal-location weighting).
Only main-role systems (rule 4) enter E3/E4; sensitivity systems are listed in E1/E2. Recovery errors use the exact
expected change of Eq. 2 (src/degradation/injection_truth.expected_for), as Table II of the paper.

  python scripts/summarize_climate_expansion.py

Inputs: results/climate_expansion/rates_rdtools.csv, rates_kappa.csv, rates_ml*.csv
Outputs: results/climate_expansion/tables/E0a_screen_by_source.csv, E0b_climate_classes.csv (screen, always),
         E1_field_rates.csv, E2_recovery_by_system.csv, E3_recovery_by_tier.csv, E4_tier_term.csv (once rates exist)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation.injection_truth import expected_for  # noqa: E402

OUT = ROOT / "results/climate_expansion"
TAB = OUT / "tables"
KEYS = ["system_id", "tier", "role", "location", "climate"]


def load() -> dict[str, pd.DataFrame]:
    def read(pattern: str) -> pd.DataFrame:
        parts = [pd.read_csv(p) for p in sorted(OUT.glob(pattern))]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ml = [pd.read_csv(p) for p in sorted(OUT.glob("rates_ml*.csv"))]
    ml = pd.concat(ml).drop_duplicates(["system_id", "model", "seed", "injection"], keep="last") if ml else pd.DataFrame()
    dedupe = lambda t: t.drop_duplicates(["system_id", "injection"], keep="last") if len(t) else t  # noqa: E731
    return {"rd": dedupe(read("rates_rdtools*.csv")), "kd": dedupe(read("rates_kappa*.csv")), "ml": ml}


def per_method(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per system, method and injection (ML: mean over seeds, with the seed SD)."""
    parts = []
    if len(runs["rd"]):
        parts.append(runs["rd"].assign(method="rdtools", seed_sd=np.nan))
    if len(runs["kd"]):
        parts.append(runs["kd"].assign(method="khd_" + runs["kd"].get("variant", "full").astype(str),
                                       seed_sd=runs["kd"].get("plr_seed_sd", np.nan)))
    if len(runs["ml"]):
        g = runs["ml"].groupby([*KEYS, "model", "injection"])
        ml = g[["plr", "t_center_years"]].mean().join(g.plr.std().rename("seed_sd")).reset_index()
        parts.append(ml.rename(columns={"model": "method"}).assign(ci_low=np.nan, ci_high=np.nan))
    if not parts:
        return pd.DataFrame(columns=[*KEYS, "method", "injection", "plr", "t_center_years"])
    cols = [*KEYS, "method", "injection", "plr", "ci_low", "ci_high", "seed_sd", "t_center_years"]
    return pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)


def recovery(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, method), g in table.groupby(["system_id", "method"]):
        base = g[g.injection == 0]
        if base.empty or not np.isfinite(base.plr.iloc[0]):
            continue
        d, tc = float(base.plr.iloc[0]), float(base.t_center_years.iloc[0])
        for r in g[g.injection != 0].itertuples():
            rows.append({**{k: getattr(r, k) for k in KEYS}, "method": method, "rate": r.injection,
                         "error": (r.plr - d) - expected_for(method, sid, r.injection, d, tc),
                         "naive_error": (r.plr - d) - r.injection})
    return pd.DataFrame(rows, columns=[*KEYS, "method", "rate", "error", "naive_error"]).dropna(subset=["error"])


def by_tier(rec: pd.DataFrame) -> pd.DataFrame:
    """Main-role systems; location means first, then the mean over locations, separately per tier."""
    main = rec[rec.role == "main"]
    if main.empty:
        return pd.DataFrame()
    loc = main.groupby(["tier", "method", "location"]).agg(
        abs_error=("error", lambda e: e.abs().mean()), bias=("error", "mean"), n_systems=("system_id", "nunique"),
        climate=("climate", "first")).reset_index()
    out = loc.groupby(["tier", "method"]).agg(
        mae_equal_location=("abs_error", "mean"), bias_equal_location=("bias", "mean"),
        worst_location_mae=("abs_error", "max"), n_locations=("location", "nunique"), n_systems=("n_systems", "sum"),
        climates=("climate", lambda c: ";".join(sorted(set(map(str, c)))))).reset_index()
    out["rank_in_tier"] = out.groupby("tier").mae_equal_location.rank(method="average")
    return out.sort_values(["tier", "mae_equal_location"])


def tier_term(rec: pd.DataFrame) -> pd.DataFrame:
    """Pooled |error| ~ method + tier, location-level observations weighted equally, cluster-robust by location.
    The tier coefficient is the S-vs-R difference the plan requires whenever the tiers are combined."""
    import statsmodels.formula.api as smf
    main = rec[rec.role == "main"]
    loc = main.groupby(["tier", "method", "location", "rate"]).error.apply(lambda e: e.abs().mean()).rename(
        "abs_error").reset_index()
    if loc.tier.nunique() < 2 or loc.location.nunique() < 3:
        return pd.DataFrame([{"term": "not estimated", "note": "needs both tiers and >= 3 locations"}])
    fit = smf.ols("abs_error ~ C(method) + C(tier, Treatment('S'))", data=loc).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(loc.location)[0]})
    table = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "se": fit.bse.values,
                          "p": fit.pvalues.values})
    table["n_obs"], table["n_locations"] = int(fit.nobs), int(loc.location.nunique())
    return table


def screen_tables() -> None:
    """E0a: candidates and outcome by tier and source; E0b: Köppen classes of the measured records (paper + passed
    expansion systems) and of the semi-synthetic weather sites, all from the 1-km raster."""
    s = pd.read_csv(OUT / "screen_tier_s.csv")
    r = pd.read_csv(OUT / "screen_tier_r.csv")
    allrows = pd.concat([s, r], ignore_index=True)
    allrows["source"] = allrows.system_id.str.extract(r"^(hk|ut|fmi|pvdaq)")[0].map(
        {"hk": "Hong Kong (Dryad)", "ut": "Utrecht (Zenodo)", "fmi": "Finland (FMI)", "pvdaq": "PVDAQ"})
    rows = []
    for (tier, source), g in allrows.groupby(["tier", "source"]):
        reasons = g[g.decision == "fail"].reason.fillna("").str.split("; ").explode()
        top = reasons[reasons != ""].str.replace(r"^< (\d) years of valid days$", r"fewer than \1 valid years", regex=True)
        rows.append({"tier": tier, "source": source, "candidates": len(g), "passed": int((g.decision == "pass").sum()),
                     "failed": int((g.decision == "fail").sum()), "pending": int((g.decision == "pending").sum()),
                     "main_failure_reason": top.value_counts().index[0] if len(top) else ""})
    pd.DataFrame(rows).to_csv(TAB / "E0a_screen_by_source.csv", index=False)
    sites = pd.read_csv(ROOT / "results/degradation_paper/tables/T0_sites.csv")
    paper = sites[sites.retained].groupby("climate").agg(paper_systems=("system_id", "size"),
                                                         paper_locations=("location", "nunique"))
    passed = allrows[allrows.decision == "pass"].groupby("climate").agg(
        expansion_systems=("system_id", "size"), expansion_locations=("location", "nunique"))
    tiers = allrows[allrows.decision == "pass"].groupby("climate").tier.agg(lambda t: "+".join(sorted(set(t))))
    semi = pd.read_csv(OUT / "semisynthetic_sites.csv").groupby("climate").site_id.agg(lambda x: ";".join(x))
    classes = pd.concat([paper, passed, tiers.rename("expansion_tier"), semi.rename("semi_synthetic_sites")],
                        axis=1).rename_axis("climate").reset_index()
    for c in ("paper_systems", "paper_locations", "expansion_systems", "expansion_locations"):
        classes[c] = classes[c].fillna(0).astype(int)
    classes["measured"] = (classes.paper_systems + classes.expansion_systems) > 0
    classes.fillna("").sort_values(["measured", "climate"], ascending=[False, True]).to_csv(
        TAB / "E0b_climate_classes.csv", index=False)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    screen_tables()
    table = per_method(load())
    if table.empty:
        print("screen tables written; no expansion rates yet (scripts/run_climate_expansion.py rd|kappa|ml)")
        return
    table[table.injection == 0].drop(columns="injection").to_csv(TAB / "E1_field_rates.csv", index=False)
    rec = recovery(table)
    rec.to_csv(TAB / "E2_recovery_by_system.csv", index=False)
    tiers = by_tier(rec)
    tiers.to_csv(TAB / "E3_recovery_by_tier.csv", index=False)
    tier_term(rec).to_csv(TAB / "E4_tier_term.csv", index=False)
    print(tiers.to_string(index=False))


if __name__ == "__main__":
    main()
