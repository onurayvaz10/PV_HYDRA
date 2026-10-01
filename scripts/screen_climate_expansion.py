"""Climate expansion, step 2: tiered screen of the candidates (reports/CLIMATE_EXPANSION_PLAN.md, rules 1-5).

  python scripts/screen_climate_expansion.py [S|R] [system ids ...]

Needs the native records (scripts/fetch_climate_expansion.py). ERA5 (Open-Meteo archive) and the Köppen-Geiger
raster are used when reachable; otherwise the affected checks are written as pending and the decision is
"pending" -- a system is never passed on an incomplete screen. No rate is computed here.
Outputs (aggregates only): results/climate_expansion/screen_tier_s.csv, screen_tier_r.csv, selection.csv,
screen_status.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import expansion as ex  # noqa: E402

OUT = ROOT / "results/climate_expansion"


def existing_main() -> dict[str, int]:
    """Systems per location already in the paper's main statistics (T0_sites, retained)."""
    sites = pd.read_csv(ROOT / "results/degradation_paper/tables/T0_sites.csv")
    return sites[sites.retained].groupby("location").size().to_dict()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tiers = [a for a in sys.argv[1:] if a in ("S", "R")] or ["S", "R"]
    pending_only = "--pending" in sys.argv          # rescreen only systems not yet decided (saves ERA5 quota)
    only = [a for a in sys.argv[1:] if a not in ("S", "R", "--pending")]
    probe = pd.read_csv(OUT / "probe_pvdaq.csv", dtype={"system_id": str}).set_index("system_id")
    for tier in tiers:
        ids = (ex.TIER_S_PARQUET + ex.TIER_S_PRIZE + ex.fmi_candidates("S")) if tier == "S" else \
            (ex.TIER_R_PARQUET + ex.TIER_R_PRIZE + ex.external_candidates())
        path = OUT / f"screen_tier_{tier.lower()}.csv"
        previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["system_id"])
        rows = []
        decided = set(previous[previous.decision.isin(["pass", "fail"])].system_id) if pending_only else set()
        for sid in ids:
            if (only and sid not in only) or ex.system_key(sid) in decided:
                continue
            if tier == "S" and sid in probe.index and not bool(probe.loc[sid, "poa"]):
                meta = ex.catalogue_row(sid)
                rows.append({"system_id": f"pvdaq_{sid}", "tier": "S", "location": meta["location"],
                             "name": meta["name"], "lat": meta["lat"], "lon": meta["lon"], "dc_kw": meta["dc_kw"],
                             "catalogue_climate": meta["catalogue_climate"], "decision": "fail",
                             "reason": "no usable on-site POA (no POA channel in the metrics metadata)"})
            else:
                try:
                    rows.append(ex.screen_system(sid))
                except Exception as exc:  # recorded, never dropped silently
                    rows.append({"system_id": ex.system_key(sid), "raw_id": sid, "tier": tier, "decision": "pending",
                                 "reason": f"screen error: {type(exc).__name__}: {exc}"})
            print({k: v for k, v in rows[-1].items() if k in ("system_id", "decision", "reason", "pending",
                                                              "valid_years", "p999_over_dc")}, flush=True)
        done = pd.DataFrame(rows)
        if not done.empty:
            table = pd.concat([previous[~previous.system_id.isin(done.system_id)], done], ignore_index=True)
            table.to_csv(path, index=False)
    frames = [pd.read_csv(OUT / f"screen_tier_{t}.csv") for t in ("s", "r") if (OUT / f"screen_tier_{t}.csv").exists()]
    screen = pd.concat(frames, ignore_index=True)
    selection = ex.select_main(screen, existing_main())
    selection.to_csv(OUT / "selection.csv", index=False)
    status = {"decisions": screen.groupby(["tier", "decision"]).size().rename("n").reset_index().to_dict("records"),
              "pending_reasons": sorted(set(p for v in screen.get("pending", pd.Series(dtype=str)).dropna()
                                            for p in str(v).split("; ") if p)),
              "not_screened_here": {k: v for k, v in ex.TIER_R_EXTERNAL.items()
                                    if k != "hongkong" or not ex.external_candidates()},
              "era5_reachable": bool((screen.get("era5", pd.Series(dtype=str)) == "ok").any()),
              "raster_reachable": bool(~screen.get("climate", pd.Series(dtype=str)).astype(str).str.startswith("pending").all()),
              "transposition": ex.TRANSPOSITION}
    (OUT / "screen_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
