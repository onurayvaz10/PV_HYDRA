"""System and split manifests of the paper.

system_manifest.csv : the 33 main measured systems with data group, location, country, Koppen-Geiger class, tier,
                      irradiance source, development-system flag, plus the excluded candidates of the long
                      sensor-grade screen with their reason.
split_manifest.csv  : per main system, the rule that defines the reference (training) period, the held-out weeks, the
                      rate period and the forward-test split (rules as implemented; dates follow from each record).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
sys.path.insert(0, str(ROOT))
from scripts.summarize_global import country_of  # noqa: E402

TAB = ROOT / "results/degradation_paper/tables"
CE = ROOT / "results/climate_expansion"
DEV = {"dkasc_7", "dkasc_12", "dkasc_14"}          # development systems of the tuning objective (CdTe, mono-Si, poly-Si)


def main() -> None:
    t0 = pd.read_csv(TAB / "T0_sites.csv")
    rows = []
    for r in t0.itertuples():
        rows.append({"system_id": r.system_id, "role": "main" if r.retained else "excluded",
                     "group": "long sensor-grade" if r.retained else "", "location": r.location,
                     "country": "United States" if r.country == "USA" else r.country, "koppen_geiger": r.climate,
                     "tier": "sensor-grade", "irradiance": "on-site POA", "technology": r.technology,
                     "dc_kw": r.dc_kw, "first": r.first, "last": r.last, "development_system": r.system_id in DEV,
                     "exclusion_reason": "" if r.retained else r.reason})
    sel = pd.read_csv(CE / "selection.csv")
    sel = sel[(sel.role == "main") & (sel.status == "final")]
    for r in sel.itertuples():
        short = r.system_id.startswith(("hk_", "ut_", "fmi_"))
        if r.system_id.startswith("hk_"):
            irr = "ERA5 GHI (on-site sensor step, Feb. 2023)"
        elif r.tier == "S":
            irr = "on-site POA"
        else:
            irr = "ERA5, transposed"
        country = country_of(r.system_id).replace("China (Hong Kong)", "Hong Kong SAR, China")
        rows.append({"system_id": r.system_id, "role": "main", "group": "short expansion" if short else "long expansion",
                     "location": r.location.replace("Hong Kong", "Hong Kong SAR, China"), "country": country,
                     "koppen_geiger": r.climate, "tier": r.tier + ("-short" if short else ""), "irradiance": irr,
                     "technology": "", "dc_kw": "", "first": "", "last": "", "development_system": False,
                     "exclusion_reason": ""})
    sm = pd.DataFrame(rows)
    out = REV
    sm.to_csv(out / "system_manifest.csv", index=False)
    main = sm[sm.role == "main"]
    assert len(main) == 33, len(main)
    sp = []
    for r in main.itertuples():
        short = r.group == "short expansion"
        sp.append({"system_id": r.system_id, "group": r.group,
                   "reference_period": "first 12 data months" if short else "first 24 data months",
                   "data_month": "calendar month with >= 150 healthy daylight hours",
                   "held_out": "every fifth ISO week of the reference period (early stopping, tuning, fit scores); "
                               "kappa-HYDRA-D: every fifth ISO week of the whole record (hourly scores)",
                   "rate_period": "whole record after the reference filters",
                   "forward_test": "train first 24 calendar months; tune next 12; test all later hours; 24-h gaps"
                                   if r.group == "long sensor-grade" else "not part of the forward test",
                   "injection": "r_inj = -0.25, -0.5, -1, -2 %/yr on the whole record"})
    pd.DataFrame(sp).to_csv(out / "split_manifest.csv", index=False)
    print(f"system_manifest.csv: {len(sm)} rows ({len(main)} main); split_manifest.csv: {len(sp)} rows")
    print(main.groupby("group").size().to_string())


if __name__ == "__main__":
    main()
