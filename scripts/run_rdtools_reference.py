"""Reference PLR (RdTools sensor workflow) for every Tier-A system.

Output: results/degradation_paper/tier_a_screen.csv, reference_plr.csv
Soiling (SRR) is estimated for hot/arid climates (B*), where it matters most.
Where no on-site temperature exists, ERA5 2-m temperature is used as ambient
temperature (recorded in the output).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation.rdtools_pipeline import gamma_for, sensor_plr
from src.degradation.tier_a import load_system, screen
from src.weather.open_meteo import reanalysis

OUT = ROOT / "results/degradation_paper"


def systems() -> list[str]:
    manifest = json.loads((ROOT / "data/raw/external/dkasc/manifest.json").read_text())
    dkasc = [f"dkasc_{v['array']}" for v in manifest.values()]
    pvdaq = ["1430", "1432", "1433", "1423", "1199", "1200", "1201", "1202", "1203", "1204", "2107", "9068"]
    available = [s for s in pvdaq if any((ROOT / "data/raw/tier_a").glob(f"*{s}_*.parquet"))]
    return dkasc + available


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    screens, results = [], []
    for sid in systems():
        try:
            frame, meta = load_system(sid)
            check = screen(frame, meta)
        except Exception as exc:
            screens.append({"system_id": sid, "pass": False, "reason": f"error: {type(exc).__name__}: {exc}"})
            traceback.print_exc()
            continue
        screens.append(check)
        print(check["system_id"], check["pass"], check.get("reason"), flush=True)
        if not check["pass"]:
            continue
        temp_basis = meta.pop("temperature_basis")
        gamma, gamma_basis = gamma_for(meta["label"])
        soiling = str(meta["climate"]).startswith("B")
        try:
            r = sensor_plr(frame, meta["dc_kw"], gamma, soiling=soiling, freq_minutes=60)
        except Exception as exc:
            r = {"status": f"error: {type(exc).__name__}: {exc}"}
            traceback.print_exc()
        results.append({**meta, **r, "gamma": gamma, "gamma_basis": gamma_basis, "temperature_basis": temp_basis})
        print("  PLR", r.get("plr"), r.get("ci_low"), r.get("ci_high"), r.get("soiling_ratio_median"), flush=True)
    pd.DataFrame(screens).to_csv(OUT / "tier_a_screen.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "reference_plr.csv", index=False)


if __name__ == "__main__":
    main()
