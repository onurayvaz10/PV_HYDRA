"""Full-record native data for Tier-A PVDAQ systems (degradation paper).

Tier A = research/commercial systems with multi-year intraday AC power and
on-site irradiance/temperature. Each system's whole archived span is read
(concurrent day-file reads); the project-defined physical screen is applied
afterwards by scripts/screen_tier_a.py. Prize systems use their documented
files (src/data/pvdaq_prize.py) with a long look-back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pvdaq_native import acquire, available_day_range
from src.data.pvdaq_prize import acquire_prize

PARQUET_SYSTEMS = ["1430", "1432", "1433", "1423", "1199", "1200", "1201", "1202", "1203", "1204"]
PRIZE_SYSTEMS = ["2107", "9068"]
OUT = ROOT / "data/raw/tier_a"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sid in PARQUET_SYSTEMS:
        try:
            span = available_day_range(sid)
            if span is None:
                summary[sid] = {"status": "no archive"}
                continue
            native, meta = acquire(sid, span[0], span[1], OUT)
            summary[sid] = {"status": "ok", "first": str(span[0].date()), "last": str(span[1].date()),
                            "rows": meta["rows"], "interval": meta["native_interval_minutes"],
                            "weather": meta["weather_channels"], "time_basis": meta.get("time_basis")}
        except Exception as exc:  # recorded, never dropped silently
            summary[sid] = {"status": f"error: {type(exc).__name__}: {exc}"}
        print(sid, summary[sid], flush=True)
    for sid in PRIZE_SYSTEMS:
        try:
            native, meta = acquire_prize(sid, 240, OUT)
            summary[sid] = {"status": "ok", "first": str(native.index.min().date()),
                            "last": str(native.index.max().date()), "rows": meta["rows"],
                            "interval": meta["native_interval_minutes"], "weather": meta["weather_channels"]}
        except Exception as exc:
            summary[sid] = {"status": f"error: {type(exc).__name__}: {exc}"}
        print(sid, summary[sid], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
    import pyarrow.fs
    pyarrow.fs.finalize_s3()
    import os
    os._exit(0)
