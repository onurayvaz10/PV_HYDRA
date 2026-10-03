"""Rank correlation between the site-level reference error and |latitude| over the 20 weather sites (exact-rate
records, technology-typical coefficient). Output: results/climate_expansion/tables/E13_latitude_spearman.csv"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/climate_expansion"


def main() -> None:
    sites = pd.read_csv(ROOT / "configs/semisynthetic_climate_sites.csv")
    runs = pd.concat([pd.read_csv(p) for p in OUT.glob("semisynthetic_climate_runs*.csv")]).dropna(subset=["plr"])
    runs = runs.drop_duplicates(["site_id", "family", "rate", "seed", "method"], keep="last")
    import sys
    sys.path.insert(0, str(ROOT))
    from src.degradation.truth_convention import first_year
    runs["error"] = runs.plr - first_year(runs.rate)          # truth at the first-year level (amendment 3)
    rows = []
    for method in ("rdtools", "khd_full"):
        r = runs[runs.method == method].groupby("site_id").error.agg(lambda e: e.abs().mean()).rename("site_mae")
        t = sites.set_index("site_id").join(r, how="inner")
        lat = t["lat"] if "lat" in t else t["latitude"]
        rho, p = spearmanr(lat.abs(), t.site_mae)
        rows.append({"method": method, "n_sites": len(t), "spearman_rho": rho, "p_value": p})
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "tables/E13_latitude_spearman.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
