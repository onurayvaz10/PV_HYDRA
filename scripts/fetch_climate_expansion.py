"""Climate expansion, step 1b: native records of the Tier-S and Tier-R PVDAQ candidates (NREL OEDI, anonymous S3).

  python scripts/fetch_climate_expansion.py [S|R] [system ids ...]
  python scripts/fetch_climate_expansion.py --selected     only the systems of results/climate_expansion/selection.csv
                                                           (+ Utrecht download and hourly build when selected)

Raw files go to data/raw/climate_expansion/ (git-ignored, never committed). Tier-S candidates without an on-site
POA channel in their metrics metadata (probe_pvdaq.csv) are not downloaded: they fail rule 1 of the Tier-S screen
whatever their data. Output (aggregates only): results/climate_expansion/fetch_summary.csv
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pvdaq_native import PREFIX, acquire  # noqa: E402
from src.data.pvdaq_prize import acquire_prize  # noqa: E402
from src.degradation import expansion as ex  # noqa: E402

OUT = ROOT / "results/climate_expansion"


def day_span(sid: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """First/last archived day, ignoring day files dated before FIRST_PLAUSIBLE_YEAR (timestamp faults)."""
    import pyarrow.fs as fs
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    years = sorted(int(m.group(1)) for info in s3.get_file_info(fs.FileSelector(PREFIX.format(sid=sid)))
                   if (m := re.search(r"year=(\d+)$", info.path)))
    dropped = sum(1 for y in years if y < ex.FIRST_PLAUSIBLE_YEAR)
    years = [y for y in years if y >= ex.FIRST_PLAUSIBLE_YEAR]
    days = []
    for year in (years[0], years[-1]):
        for info in s3.get_file_info(fs.FileSelector(PREFIX.format(sid=sid) + f"year={year}/", recursive=True)):
            match = re.search(r"year=(\d+)/month=(\d+)/day=(\d+)/", info.path)
            if info.type == fs.FileType.File and match:
                days.append(pd.Timestamp(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return min(days), max(days), dropped


def fetch(sid: str) -> dict:
    row = {"system_id": f"pvdaq_{sid}", "tier": ex.tier_of(sid)}
    started = time.perf_counter()
    if sid in ex.PRIZE:
        native, meta = acquire_prize(sid, ex.PRIZE_MONTHS, ex.RAW)
        row.update({"source": "2023 Solar Data Prize files", "first": str(native.index.min().date()),
                    "last": str(native.index.max().date()), "power_channel": "prize_documented_units",
                    "files": "; ".join(f for files in meta["files"].values() for f in files)})
    else:
        first, last, dropped = day_span(sid)
        native, meta = acquire(sid, first, last, ex.RAW, case_insensitive_weather=True)
        row.update({"source": "PVDAQ parquet archive", "first": str(first.date()), "last": str(last.date()),
                    "pre_1990_year_folders_ignored": dropped, "power_channel": meta["metric_rule"],
                    "power_metrics": ";".join(meta["power_metrics"]), "power_units": ";".join(meta["power_units"]),
                    "time_basis": meta["time_basis"]})
    row.update({"status": "ok", "rows": meta["rows"], "native_interval_min": meta["native_interval_minutes"],
                "weather_channels": ";".join(meta["weather_channels"]), "cache_sha256": meta["sha256"],
                "seconds": round(time.perf_counter() - started, 1)})
    return row


UTRECHT_FILE = "https://zenodo.org/api/records/10953360/files/unfiltered_pv_power_measurements.csv/content"


def fetch_utrecht() -> None:
    """Unfiltered 1-min Utrecht power file (4.8 GB, Zenodo 10953360, CC BY 4.0) -> hourly complete-hour means."""
    import requests
    folder = ROOT / "data/raw/external/utrecht"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "unfiltered_pv_power_measurements.csv"
    if not target.exists():
        tmp = target.with_suffix(".part")
        with requests.get(UTRECHT_FILE, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
        tmp.rename(target)
    for name in ("metadata.csv", "metadata_units.csv"):
        if not (folder / name).exists():
            url = f"https://zenodo.org/api/records/10953360/files/{name}/content"
            (folder / name).write_bytes(requests.get(url, timeout=120).content)
    if not ex.UTRECHT_HOURLY.exists():
        ex.build_utrecht_hourly()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--selected" in sys.argv:
        sel = pd.read_csv(OUT / "selection.csv")
        raw = sel.raw_id.astype(str).tolist()
        if any(r.startswith("utrecht:") for r in raw):
            fetch_utrecht()
        sys.argv = [a for a in sys.argv if a != "--selected"] + [r for r in raw if ":" not in r]
        if not [a for a in sys.argv[1:] if a not in ("S", "R")]:
            return
    tiers = [a for a in sys.argv[1:] if a in ("S", "R")] or ["S", "R"]
    only = [a for a in sys.argv[1:] if a not in ("S", "R")]
    probe = pd.read_csv(OUT / "probe_pvdaq.csv", dtype={"system_id": str}).set_index("system_id")
    todo = []
    if "R" in tiers:
        todo += ex.TIER_R_PARQUET + ex.TIER_R_PRIZE
    if "S" in tiers:
        todo += ex.TIER_S_PARQUET + ex.TIER_S_PRIZE
    summary_file = OUT / "fetch_summary.csv"
    previous = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame(columns=["system_id"])
    rows = []
    for sid in todo:
        if only and sid not in only:
            continue
        if sid in probe.index and not bool(probe.loc[sid, "poa"]) and ex.tier_of(sid) == "S":
            rows.append({"system_id": f"pvdaq_{sid}", "tier": "S", "status": "not downloaded: no on-site POA channel"})
            continue
        try:
            rows.append(fetch(sid))
        except Exception as exc:  # recorded, never dropped silently
            rows.append({"system_id": f"pvdaq_{sid}", "tier": ex.tier_of(sid), "status": f"error: {type(exc).__name__}: {exc}"})
        print(rows[-1], flush=True)
        done = pd.DataFrame(rows)
        pd.concat([previous[~previous.system_id.isin(done.system_id)], done]).to_csv(summary_file, index=False)
    done = pd.DataFrame(rows)
    pd.concat([previous[~previous.system_id.isin(done.system_id)], done]).to_csv(summary_file, index=False)


if __name__ == "__main__":
    main()
    import pyarrow.fs
    pyarrow.fs.finalize_s3()
    import os
    os._exit(0)
