"""Site table of all 24 candidate systems: name, location, country, coordinates, elevation, climate class,
mounting, technology, capacity, logging interval and analysis period.

Sources (nothing typed by hand except the DKASC site record used throughout the analysis):
  PVDAQ   names, locations, coordinates, elevation, capacity, mounting: NREL PVDAQ catalogue
          (data/reference/pvdaq/pvdaq_systems_20250729.csv); logging interval: download records (data/raw/tier_a/*.json)
  DKASC   array label (manufacturer, rating, technology, mounting, year): DKASC listing (data/raw/external/dkasc/manifest.json);
          site coordinates and elevation: the DKASC site record of src/data/external_sources.py, as used for ERA5 and
          clear-sky calculations
  Climate Koppen-Geiger class from the 1-km 1991-2020 map of Beck et al. (2023) at each coordinate
          (data/reference/climate); the PVDAQ catalogue class is kept for comparison
  Period  first and last timestamp after screening (results/degradation_paper/tier_a_screen.csv)
Output: results/degradation_paper/tables/T0_sites.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.climate.classify import classify_sites, verify_reference  # noqa: E402
from src.data.external_sources import dkasc  # noqa: E402,F401  (site record below mirrors it)

OUT = ROOT / "results/degradation_paper/tables/T0_sites.csv"
DKASC_SITE = {"location": "Alice Springs, NT", "country": "Australia", "latitude": -23.7618, "longitude": 133.8748,
              "elevation_m": 546.0}                        # = external_sources.dkasc() site record
STATE_COUNTRY = "USA"


def main() -> None:
    screen = pd.read_csv(ROOT / "results/degradation_paper/tier_a_screen.csv")
    cat = pd.read_csv(ROOT / "data/reference/pvdaq/pvdaq_systems_20250729.csv").drop_duplicates("system_id")
    manifest = json.loads((ROOT / "data/raw/external/dkasc/manifest.json").read_text(encoding="utf-8"))
    by_array = {f"dkasc_{v['array']}": v for v in manifest.values()}
    intervals = {}
    for f in (ROOT / "data/raw/tier_a").glob("*.json"):
        info = json.loads(f.read_text(encoding="utf-8"))
        if "system_id" in info:
            intervals[str(info["system_id"])] = info.get("native_interval_minutes")
    rows = []
    for s in screen.itertuples():
        sid = s.system_id
        if sid.startswith("dkasc_"):
            v = by_array[sid]
            parts = [x.strip() for x in v["label"].split(",")]
            rows.append({"system_id": sid, "source": "DKASC", "name": f"Array {v['array']}: {v['label']}",
                         **DKASC_SITE, "catalogue_climate": "", "mounting": parts[3].lower() if len(parts) > 3 else "",
                         "technology": parts[2], "manufacturer": parts[0], "installed": parts[4] if len(parts) > 4 else "",
                         "dc_kw": float(v["dc_kw"]), "interval_min": 5})
        else:
            num = int(sid.split("_")[1])
            c = cat[cat.system_id == num].iloc[0]
            mount = "single-axis tracker" if c.tracking == "tracking" else f"fixed, {c.type}" if c.type != "unknown" else "fixed"
            rows.append({"system_id": sid, "source": "PVDAQ", "name": str(c.system_public_name).split("] ")[-1],
                         "location": c.site_location, "country": STATE_COUNTRY, "latitude": float(c.latitude),
                         "longitude": float(c.longitude), "elevation_m": float(c.elevation_m),
                         "catalogue_climate": str(c.kg_climate).upper().replace("BWH", "BWh").replace("BSK", "BSk")
                         .replace("CFA", "Cfa").replace("CSA", "Csa"),
                         "mounting": mount, "technology": "not reported", "manufacturer": "", "installed": "",
                         "dc_kw": float(c.dc_capacity_kW), "interval_min": intervals.get(str(num))})
        rows[-1].update({"first": str(s.first)[:10], "last": str(s.last)[:10], "retained": bool(s._13),
                         "reason": "" if pd.isna(s.reason) else s.reason})
    t = pd.DataFrame(rows)
    raster, codebook, _ = verify_reference(ROOT / "data/reference/climate/source.json")
    k = classify_sites(t[["system_id", "latitude", "longitude"]], raster, codebook).set_index("system_id").climate_class
    t.insert(8, "climate", t.system_id.map(k))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(t[["system_id", "name", "location", "latitude", "longitude", "elevation_m", "climate", "catalogue_climate",
             "mounting", "technology", "dc_kw", "interval_min", "first", "last", "retained"]].to_string())


if __name__ == "__main__":
    main()
