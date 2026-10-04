"""Tier-S screen of the US DOE RTC baseline systems (amendment 4), before any rate is computed.

The screen is expansion.screen_system unchanged (clock convention against ERA5, power scale, night offset, POA
saturation, >= 4 valid years, two-reference POA drift). The Nevada inverters are excluded a priori (same site and
start date as PVDAQ 1423, already in the main benchmark); they are screened only to record the overlap.
Output: results/fixed_mask_rtc/rtc_screen.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import expansion as ex  # noqa: E402


def main() -> None:
    meta = ex.rtc_metadata()
    rows = []
    for _, m in meta.iterrows():
        row = ex.screen_system(f"rtc:{m.rcid}")
        row["site"] = m.invtsite
        row["included"] = bool(row.get("decision") == "pass" and m.invtsite != "nevada")
        row["exclusion"] = "a priori: same site and start date as PVDAQ 1423" if m.invtsite == "nevada" else ""
        # amendment 4b: fails only the >= 4 y record-length rule and has >= 3 valid years -> exploratory extension
        reasons = [r for r in str(row.get("reason") or "").split("; ") if r and r != "nan"]
        row["exploratory"] = bool(m.invtsite != "nevada" and row.get("decision") == "fail"
                                  and reasons == ["< 4 years of valid days"] and float(row.get("valid_years", 0)) >= 3
                                  and not row.get("pending"))
        rows.append(row)
        print({k: row.get(k) for k in ("system_id", "location", "climate", "decision", "reason", "valid_years",
                                       "time_convention", "poa_era5_ratio_trend_per_year",
                                       "poa_clearsky_envelope_trend_per_year", "included")}, flush=True)
    out = ROOT / "results/fixed_mask_rtc"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rtc_screen.csv", index=False)


if __name__ == "__main__":
    main()
