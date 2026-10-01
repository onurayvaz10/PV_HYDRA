"""Rank tests for the climate expansion (descriptive support for E6/E8).

Known-truth weather sites: Friedman test over the 20 sites (one per country; block = site, value = the site's mean
absolute error over responses, rates and seeds) and the number of sites where each method is most accurate.
Measured records: Friedman test over countries (block = country, value = country mean absolute error, E5 averaged
over the country's groups) and the per-country winner. With 5 countries the measured test has little power and is
reported as descriptive.

Output: results/climate_expansion/tables/E11_rank_tests.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results/climate_expansion/tables"


def friedman(wide: pd.DataFrame, label: str) -> list[dict]:
    wide = wide.dropna()
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    ranks = wide.rank(axis=1).mean()
    best = wide.idxmin(axis=1).value_counts()
    rows = [{"set": label, "method": m, "mean_rank": float(ranks[m]), "n_blocks": len(wide),
             "blocks_best": int(best.get(m, 0)), "friedman_chi2": float(stat), "friedman_p": float(p)}
            for m in wide.columns]
    return rows


def main() -> None:
    e7 = pd.read_csv(TAB / "E7_semisynthetic_by_site.csv")
    semi = e7.pivot_table(index="site_id", columns="method", values="mae")
    e5 = pd.read_csv(TAB / "E5_country_recovery.csv")
    meas = e5.groupby(["country", "method"]).mae.mean().unstack()
    out = pd.DataFrame(friedman(semi, "known-truth sites (20 countries)") + friedman(meas, "measured (5 countries)"))
    out.sort_values(["set", "mean_rank"]).to_csv(TAB / "E11_rank_tests.csv", index=False)
    print(out.sort_values(["set", "mean_rank"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
