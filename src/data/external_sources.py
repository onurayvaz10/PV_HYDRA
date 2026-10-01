"""Loaders that bring open measured-PV datasets to one native format.

Every loader returns ``(native, site)``:
  native : DataFrame indexed by raw time stamps (naive) with ``power_w`` and,
           when measured on site, ``onsite_ghi_wm2`` / ``onsite_poa_wm2`` /
           ``onsite_ambient_temp_c``. Values are exactly as published (units
           converted only as documented); gaps stay gaps.
  site   : dict with system_id, source, latitude, longitude, elevation_m,
           tz_name, tilt, azimuth, tracking, dc_capacity_kw, capacity_basis,
           native_interval_minutes, climate-independent descriptors.
The time-stamp convention (UTC / local standard / local DST) is *not* assumed
here; the pre-specified screen detects it from clear-day alignment.

Sources (licences verified 2026-09-28):
  hongkong : Dryad doi:10.5061/dryad.m37pvmd99 (CC0); Sci Data 2025, doi:10.1038/s41597-025-04397-y
  pvod     : Science Data Bank, doi:10.11922/sciencedb.01094 (CC BY 4.0);
             Yao et al. 2021, doi:10.1016/j.solener.2021.09.050. Time stamps UTC
             (dataset page). Power column in MW (station capacity in kW).
  utrecht  : Zenodo 10953360 (CC BY 4.0); Visser et al. 2022, doi:10.1063/5.0100939
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data/raw/external"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("centre", "center").replace("sports", "sport"))


# ------------------------------------------------------------------ Hong Kong
def hongkong_catalogue() -> pd.DataFrame:
    z = zipfile.ZipFile(EXTERNAL / "hongkong/dataset.zip")
    ttl = z.read("Dataset/Metadata/PV generation system metadata.ttl").decode("utf-8", "replace")
    rows = []
    for block in re.split(r"\n(?=pvsystem:\S+ a brick:PV_Generation_System)", ttl):
        head = re.match(r"pvsystem:(\S+) a brick:PV_Generation_System", block)
        if not head:
            continue
        get = lambda pattern: (m.group(1) if (m := re.search(pattern, block, re.S)) else None)  # noqa: E731
        rows.append({"name": head.group(1),
                     "latitude": float(get(r"brick:latitude ([\d.\-]+)")),
                     "longitude": float(get(r"brick:longitude ([\d.\-]+)")),
                     "elevation_m": float(get(r"ext:altitude \[ brick:hasUnit unit:M ;\s*brick:value ([\d.\-]+)") or 0),
                     "rated_kw": float(get(r"ext:ratedPowerOutput \[ brick:hasUnit unit:KW ;\s*brick:value ([\d.\-]+)") or "nan"),
                     "connection_date": get(r'ext:connectionDate \[ brick:value "([^"]+)"')})
    catalogue = pd.DataFrame(rows)
    files = [n for n in z.namelist() if "Site level dataset" in n and n.endswith(".csv")]
    by_norm = {_norm(Path(f).stem): f for f in files}
    catalogue["file"] = [by_norm.get(_norm(name)) for name in catalogue.name]
    return catalogue


def hongkong_weather() -> pd.DataFrame:
    z = zipfile.ZipFile(EXTERNAL / "hongkong/dataset.zip")
    parts = {}
    for folder, column, name in (("Irradiance", "Irradiance (W/m2)", "onsite_ghi_wm2"),
                                 ("Temperature", None, "onsite_ambient_temp_c")):
        frames = []
        for year in (2021, 2022, 2023):
            path = f"Dataset/Time series dataset/Meteorological dataset/{folder}/{folder}_{year}.csv"
            if path in z.namelist():
                frame = pd.read_csv(io.BytesIO(z.read(path)))
                frame["Time"] = pd.to_datetime(frame.Time, errors="coerce")
                value = column if column else frame.columns[1]
                frames.append(frame.set_index("Time")[value].rename(name))
        if frames:
            parts[name] = pd.concat(frames).sort_index()
    weather = pd.DataFrame(parts)
    return weather[~weather.index.duplicated()]


def hongkong(system: str, weather: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    catalogue = hongkong_catalogue().set_index("name")
    row = catalogue.loc[system]
    z = zipfile.ZipFile(EXTERNAL / "hongkong/dataset.zip")
    data = pd.read_csv(io.BytesIO(z.read(row.file)))
    data["Time"] = pd.to_datetime(data.Time, errors="coerce")
    native = data.dropna(subset=["Time"]).set_index("Time")[["power(W)"]].rename(columns={"power(W)": "power_w"})
    native = native[~native.index.duplicated()].sort_index()
    weather = hongkong_weather() if weather is None else weather
    step = int(round(native.index.to_series().diff().dt.total_seconds().median() / 60))
    # 1-min site weather averaged over the same interval-start bins as power.
    binned = weather.groupby(weather.index.floor(f"{step}min")).mean()
    native = native.join(binned, how="left")
    site = {"system_id": f"hk_{_norm(system)[:24]}", "source": "hongkong", "name": system,
            "latitude": row.latitude, "longitude": row.longitude, "elevation_m": row.elevation_m,
            "tz_name": "Asia/Hong_Kong", "tilt": 0.0, "azimuth": 180.0, "tracking": "fixed",
            "orientation_basis": "unknown; horizontal envelope approximation",
            "dc_capacity_kw": row.rated_kw, "capacity_basis": "rated power output (metadata)",
            "native_interval_minutes": step, "record_start": str(row.connection_date)}
    return native, site


# ---------------------------------------------------------------------- PVOD
def pvod_catalogue() -> pd.DataFrame:
    z = zipfile.ZipFile(EXTERNAL / "pvod/PVODdatasets_v1.0.zip")
    meta = pd.read_csv(io.BytesIO(z.read("metadata.csv")))
    meta["tilt"] = meta.Array_Tilt.str.extract(r"(\d+)").astype(float)
    return meta


def pvod(station: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns native (power + local measurements), NWP frame, site."""
    meta = pvod_catalogue().set_index("Station_ID").loc[station]
    z = zipfile.ZipFile(EXTERNAL / "pvod/PVODdatasets_v1.0.zip")
    data = pd.read_csv(io.BytesIO(z.read(f"{station}.csv")))
    data["date_time"] = pd.to_datetime(data.date_time)
    data = data.set_index("date_time").sort_index()
    native = pd.DataFrame({"power_w": data.power * 1e6,                     # published in MW
                           "onsite_ghi_wm2": data.lmd_totalirrad,
                           "onsite_ambient_temp_c": data.lmd_temperature})
    nwp = data[[c for c in data.columns if c.startswith("nwp_")]]
    site = {"system_id": f"pvod_{station}", "source": "pvod", "name": station,
            "latitude": float(meta.Latitude), "longitude": float(meta.Longitude), "elevation_m": 0.0,
            "tz_name": "Asia/Shanghai", "tilt": float(meta.tilt), "azimuth": 180.0, "tracking": "fixed",
            "dc_capacity_kw": float(meta.Capacity), "capacity_basis": "station capacity (metadata, kW)",
            "native_interval_minutes": 15, "time_basis": "UTC (dataset documentation)"}
    return native, nwp, site


# ------------------------------------------------------------------- Utrecht
def utrecht_catalogue() -> pd.DataFrame:
    meta = pd.read_csv(EXTERNAL / "utrecht/metadata.csv", sep=";").dropna(axis=1, how="all")
    meta["latitude"] = (meta.north + meta.south) / 2
    meta["longitude"] = (meta.west + meta.east) / 2
    meta["span_days"] = (pd.to_datetime(meta.end_ts) - pd.to_datetime(meta.begin_ts)).dt.days
    return meta


def utrecht_site(row) -> dict:
    return {"system_id": f"ut_{row.ID}", "source": "utrecht", "name": row.ID,
            "latitude": float(row.latitude), "longitude": float(row.longitude), "elevation_m": 0.0,
            "tz_name": "Europe/Amsterdam", "tilt": float(row.tilt), "azimuth": float(row.azimuth),
            "tracking": "fixed", "dc_capacity_kw": float(row.estimated_dc_capacity) / 1000,
            "capacity_basis": "estimated DC capacity (published metadata, W -> kW)",
            "native_interval_minutes": 1, "time_basis": "UTC (+00:00 in published time stamps)"}


# --------------------------------------------------------------------- DKASC
DKASC_TERMS = ("Desert Knowledge Australia Solar Centre data; research use with citation and the DKA "
               "disclaimer; no redistribution of > 5,000 data cells without consent")


def dkasc(file_name: str) -> tuple[pd.DataFrame, dict]:
    """Alice Springs array: Active_Power (kW), on-site GHI / tilted irradiance / temperature."""
    import json as _json
    manifest = _json.loads((EXTERNAL / "dkasc/manifest.json").read_text())
    info = manifest[file_name]
    # DKASC exports are cut server-side at ~280-300 MB, sometimes mid-line: keep
    # only complete lines (the listed file size equals the downloaded size).
    raw = (EXTERNAL / "dkasc" / file_name).read_bytes()
    raw = raw[: raw.rfind(b"\n") + 1]
    data = pd.read_csv(io.BytesIO(raw),
                       usecols=["timestamp", "Active_Power", "Global_Horizontal_Radiation",
                                "Radiation_Global_Tilted", "Weather_Temperature_Celsius"])
    data["timestamp"] = pd.to_datetime(data.timestamp, errors="coerce")
    data = data.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    data = data[~data.index.duplicated()]
    native = pd.DataFrame({"power_w": pd.to_numeric(data.Active_Power, errors="coerce") * 1000,
                           "onsite_ghi_wm2": pd.to_numeric(data.Global_Horizontal_Radiation, errors="coerce"),
                           "onsite_poa_wm2": pd.to_numeric(data.Radiation_Global_Tilted, errors="coerce"),
                           "onsite_ambient_temp_c": pd.to_numeric(data.Weather_Temperature_Celsius, errors="coerce")})
    native = native.dropna(subset=["power_w"])
    site = {"system_id": f"dkasc_{info['array']}", "source": "dkasc", "name": info["label"],
            "latitude": -23.7618, "longitude": 133.8748, "elevation_m": 546.0,
            "tz_name": "Australia/Darwin", "tilt": 20.0, "azimuth": 0.0, "tracking": "fixed",
            "orientation_basis": "north-facing 20 deg assumed for DKASC fixed arrays; clear-day rule validates",
            "dc_capacity_kw": float(info["dc_kw"]), "capacity_basis": "array rating (DKASC listing)",
            "native_interval_minutes": 5, "terms": DKASC_TERMS}
    return native, site