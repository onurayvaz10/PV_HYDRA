"""Climate expansion, step 5: global (country-weighted) summaries over the paper's systems and the expansion systems.

  python scripts/summarize_global.py

Countries are the independent units (one value per country and method: the mean over the country's systems of the mean
|error| over the injected rates), because the measured records come from few countries and several systems share one.
Short-record systems (Hong Kong, Utrecht, Finland: 2-3 valid years, 12-month reference) form a separate group and are
never pooled with the long-record groups without the group label (plan, amendments of 2026-10-01).

Inputs: results/degradation_paper/tables/T2_recovery_by_system.csv, T0_sites.csv (paper, 15 systems),
        results/climate_expansion/tables/E2_recovery_by_system.csv, E1_field_rates.csv, selection.csv (expansion),
        results/climate_expansion/semisynthetic_climate_runs*.csv, semisynthetic_sites.csv (known-truth sites).
Outputs: results/climate_expansion/tables/E5_country_recovery.csv, E6_group_recovery.csv, E7_semisynthetic_by_site.csv,
         E8_semisynthetic_by_continent.csv, E9_method_spread.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = RES / "climate_expansion"
TAB = OUT / "tables"
PAPER = RES / "degradation_paper/tables"

CONTINENT = {"Australia": "Oceania", "New Zealand": "Oceania", "United States": "North America", "Netherlands": "Europe",
             "Finland": "Europe", "Spain": "Europe", "China (Hong Kong)": "Asia", "Singapore": "Asia", "Japan": "Asia",
             "India": "Asia", "United Arab Emirates": "Asia", "Turkey": "Asia", "Mongolia": "Asia", "Morocco": "Africa",
             "Kenya": "Africa", "Nigeria": "Africa", "South Africa": "Africa", "Brazil": "South America",
             "Chile": "South America", "Argentina": "South America"}
ML = ["lstm", "gru", "tcn", "patchtst", "informer", "xgboost", "random_forest"]


def country_of(system_id: str) -> str:
    if system_id.startswith("dkasc_"):
        return "Australia"
    if system_id.startswith("hk_"):
        return "China (Hong Kong)"
    if system_id.startswith("ut_"):
        return "Netherlands"
    if system_id.startswith("fmi_"):
        return "Finland"
    return "United States"


def group_of(system_id: str) -> str:
    if system_id.startswith(("hk_", "ut_", "fmi_")):
        return "short-record (2-3 y)"
    if system_id.startswith(("dkasc_", )) or system_id in ("pvdaq_1430", "pvdaq_1423", "pvdaq_2107"):
        return "paper (long record)"
    return "expansion PVDAQ (3-17 y)"


def measured() -> pd.DataFrame:
    paper = pd.read_csv(PAPER / "T2_recovery_by_system.csv")
    paper = paper[paper.method.isin(["rdtools", "khd_full", *ML])]
    rows = [paper[["system_id", "method", "rate", "error"]]]
    e2 = OUT / "tables/E2_recovery_by_system.csv"
    if e2.exists():
        e = pd.read_csv(e2)
        e = e[(e.role == "main") & e.method.isin(["rdtools", "khd_full", *ML])]
        rows.append(e[["system_id", "method", "rate", "error"]])
    table = pd.concat(rows, ignore_index=True)
    table["country"] = table.system_id.map(country_of)
    table["group"] = table.system_id.map(group_of)
    table["abs_error"] = table.error.abs()
    return table


def country_tables(table: pd.DataFrame) -> None:
    system = table.groupby(["group", "country", "method", "system_id"]).abs_error.mean().reset_index()
    country = system.groupby(["group", "country", "method"]).agg(mae=("abs_error", "mean"), n_systems=("system_id", "nunique")
                                                                  ).reset_index()
    country.to_csv(TAB / "E5_country_recovery.csv", index=False)
    grp = country.groupby(["group", "method"]).agg(mae_equal_country=("mae", "mean"), n_countries=("country", "nunique"),
                                                   n_systems=("n_systems", "sum")).reset_index()
    grp["rank_in_group"] = grp.groupby("group").mae_equal_country.rank(method="average")
    allc = country.groupby(["country", "method"]).mae.mean().reset_index()      # one value per country over its groups
    overall = allc.groupby("method").mae.agg(["mean", "count"]).rename(
        columns={"mean": "mae_equal_country", "count": "n_countries"}).reset_index()
    overall["group"] = "all measured (country-weighted)"
    overall["rank_in_group"] = overall.mae_equal_country.rank(method="average")
    pd.concat([grp, overall], ignore_index=True).sort_values(["group", "mae_equal_country"]).to_csv(
        TAB / "E6_group_recovery.csv", index=False)


def semisynthetic() -> None:
    parts = [pd.read_csv(p) for p in sorted(OUT.glob("semisynthetic_climate_runs*.csv"))]
    if not parts:
        return
    runs = pd.concat(parts).dropna(subset=["site_id", "method", "error"]).drop_duplicates(
        ["site_id", "family", "rate", "seed", "method"], keep="last")
    sites = pd.read_csv(OUT / "semisynthetic_sites.csv")[["site_id", "country", "climate"]]
    runs = runs.drop(columns=[c for c in ("climate",) if c in runs]).merge(sites, on="site_id", how="left")
    runs["continent"] = runs.country.map(CONTINENT)
    runs["abs_error"] = runs.error.abs()
    by_site = runs.groupby(["site_id", "country", "climate", "method"]).agg(
        mae=("abs_error", "mean"), bias=("error", "mean"), n=("error", "size")).reset_index()
    by_site.to_csv(TAB / "E7_semisynthetic_by_site.csv", index=False)
    by_site["continent"] = by_site.country.map(CONTINENT)
    by_cont = by_site.groupby(["continent", "method"]).agg(mae=("mae", "mean"), n_sites=("site_id", "nunique")).reset_index()
    overall = by_site.groupby("method").agg(mae=("mae", "mean"), worst_site=("mae", "max"),
                                            n_sites=("site_id", "nunique"), n_classes=("climate", "nunique")).reset_index()
    overall.insert(0, "continent", "all (one site per country)")
    pd.concat([by_cont, overall], ignore_index=True).to_csv(TAB / "E8_semisynthetic_by_continent.csv", index=False)


def spread() -> None:
    """Disagreement among the seven ML models and kappa-HYDRA-D on the field rate of the same system (E1)."""
    f = OUT / "tables/E1_field_rates.csv"
    if not f.exists():
        return
    e1 = pd.read_csv(f)
    e1 = e1[e1.method.isin([*ML, "khd_full"])]
    g = e1.groupby("system_id").plr.agg(lambda s: float(s.max() - s.min()))
    out = pd.DataFrame({"system_id": g.index, "range_pct_per_year": g.values})
    out["country"] = out.system_id.map(country_of)
    out["group"] = out.system_id.map(group_of)
    out.to_csv(TAB / "E9_method_spread.csv", index=False)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    table = measured()
    country_tables(table)
    semisynthetic()
    spread()
    print(pd.read_csv(TAB / "E6_group_recovery.csv").to_string(index=False))


if __name__ == "__main__":
    main()
