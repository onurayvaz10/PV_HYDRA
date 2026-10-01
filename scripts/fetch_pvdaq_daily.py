"""Daily energy of long-record PVDAQ residential systems (degradation module).

Pre-specified selection (fixed before download): catalogue systems with
>= 8 years, qa_status == pass, DC capacity > 0, tilt and azimuth known, named
IANA time zone, climate class from the Beck et al. (2023) raster; per class
the up-to-8 longest records. The residential archive publishes daily
aggregates only (daily max/mean power, daily energy), which is what the
weather-normalised year-on-year degradation analysis needs.
Output: data/raw/external/pvdaq_daily/<system_id>.parquet (+ selection CSV).
"""

from __future__ import annotations

import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.fs as fs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.climate.classify import classify_sites, verify_reference

OUT = ROOT / "data/raw/external/pvdaq_daily"
PREFIX = "oedi-data-lake/pvdaq/csv/pvdata/system_id={sid}/"


def selection() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data/reference/pvdaq/pvdaq_systems_20250729.csv").drop_duplicates("system_id")
    s = s.dropna(subset=["latitude", "longitude"])
    keep = ((s.years >= 8) & s.qa_status.eq("pass") & (s.dc_capacity_kW > 0) & s.tilt.notna()
            & s.azimuth.notna() & s.timezone_or_utc_offset.fillna("").str.contains("/") & (s.system_id >= 10000))
    s = s.loc[keep].copy()
    raster, codebook, _ = verify_reference(ROOT / "data/reference/climate/source.json")
    s = classify_sites(s, raster, codebook).dropna(subset=["climate_class"])
    return s.sort_values(["climate_class", "years"], ascending=[True, False]).groupby("climate_class").head(8)


def fetch(s3, sid: int) -> pd.DataFrame | None:
    infos = s3.get_file_info(fs.FileSelector(PREFIX.format(sid=sid), recursive=True))
    parts = []
    for info in infos:
        if info.type == fs.FileType.File and info.path.endswith(".csv"):
            with s3.open_input_stream(info.path) as handle:
                frame = pd.read_csv(io.BytesIO(handle.read()))
            frame = frame.rename(columns={frame.columns[0]: "date"})
            energy = [c for c in frame.columns if re.search(r"ac_energy.*daily_sum", c)]
            if not energy:
                continue
            frame["energy_kwh"] = frame[energy].sum(axis=1, min_count=len(energy))
            parts.append(frame[["date", "energy_kwh"]])
    if not parts:
        return None
    daily = pd.concat(parts)
    daily["date"] = pd.to_datetime(daily.date, errors="coerce")
    return daily.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = selection()
    chosen.to_csv(OUT / "selection.csv", index=False)
    print({"systems": len(chosen), "classes": int(chosen.climate_class.nunique())}, flush=True)
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")

    def work(sid: int):
        path = OUT / f"{sid}.parquet"
        if path.exists():
            return sid, "cached"
        daily = fetch(s3, sid)
        if daily is None:
            return sid, "no energy column"
        daily.to_parquet(path, index=False)
        return sid, len(daily)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for sid, status in pool.map(work, chosen.system_id.astype(int)):
            print(sid, status, flush=True)
    fs.finalize_s3()


if __name__ == "__main__":
    main()
