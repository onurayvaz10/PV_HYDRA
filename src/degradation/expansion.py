"""Climate expansion (reports/CLIMATE_EXPANSION_PLAN.md): candidates, loaders and the tiered screen.

Tier S  on-site plane-of-array irradiance: the Tier-A screen (tier_a.screen) unchanged, 4 years of valid days.
Tier R  no on-site POA: ERA5 GHI/DNI/DHI (Open-Meteo archive) transposed to the published tilt/azimuth
        (pvlib Perez 1990, albedo 0.2, solar position at the hour midpoint); 3 years of valid days.
        Where an on-site GHI sensor exists (Maui 2105; orientation "varies") the GHI itself is the irradiance
        input (horizontal approximation, as for Hong Kong) and the two-reference drift rule is applied to it.

Every ERA5- or raster-dependent quantity is computed only when its source is reachable; otherwise it is
recorded as pending and the system's decision stays "pending" -- never guessed. Rules that do not depend on
ERA5 (power scale, night offset on the sensor POA, POA saturation, Tier-S record length) are final.
Operational details were fixed on 2026-09-30 before any rate of a new system (plan, "Operational
specification"); a provisional clock convention from the clear-sky model is a diagnostic only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.degradation import tier_a

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/climate_expansion"
CATALOGUE = ROOT / "data/reference/pvdaq/pvdaq_systems_20250729.csv"

# Candidates exactly as fixed in the plan (2026-09-30).
TIER_S_PARQUET = ["4", "10", "33", "50", "51", "1208", "1283", "1289", "1332", "1239", "34", "35", "1276", "1277",
                  "1278", "1367", "1368", "1369"]
TIER_S_PRIZE = ["9069", "9068"]
TIER_R_PARQUET = ["1246", "1255", "1272", "1217", "1221", "1225"]
TIER_R_PRIZE = ["2105"]
TIER_R_EXTERNAL = {"hongkong": "Hong Kong stations, Dryad doi:10.5061/dryad.m37pvmd99", "utrecht": "Utrecht systems, Zenodo 10953360"}
PRIZE = set(TIER_S_PRIZE) | set(TIER_R_PRIZE)
PRIZE_MONTHS = 480                                   # whole record (prize files start 2016-2018)
FIRST_PLAUSIBLE_YEAR = 1990                          # system 50 carries day files dated 1822 (timestamp fault)
TZ = {"9068": "America/Denver", "9069": "America/New_York", "2105": "Pacific/Honolulu"}
# Some catalogue rows give hours west of UTC instead of a zone name (Golden 7, Social Circle 5, Maui 10); the US
# zone of that offset is used, so the clock hypotheses (standard, DST, UTC) are the same as for named zones.
TZ_CODES = {"5": "America/New_York", "6": "America/Chicago", "7": "America/Denver", "8": "America/Los_Angeles",
            "10": "Pacific/Honolulu"}
PRIZE_ORIENTATION = {"9069": (20.0, 180.0), "9068": (np.nan, np.nan), "2105": (np.nan, np.nan)}  # metadata JSON
MIN_YEARS = {"S": 4.0, "R": 3.0}
MIN_YEARS_R_SHORT = 2.0          # amendment 2026-10-01: Hong Kong and Utrecht records (2-3 y); YoY needs >= 2 y; reported as R-short
MAX_PER_LOCATION = 3
TRANSPOSITION = {"model": "perez", "albedo": 0.2}


def tier_of(sid: str) -> str:
    if sid.startswith("fmi:"):
        return FMI_SITES[sid.split(":", 1)[1]]["tier"]
    return "S" if sid in TIER_S_PARQUET or sid in TIER_S_PRIZE else "R"


def external_candidates() -> list[str]:
    """Tier-R stations of the Zenodo datasets present under data/raw/external (keys 'hongkong:<station>',
    'utrecht:<ID>'). All Hong Kong stations form one location, 'Hong Kong'; all Utrecht systems 'Utrecht' (rule 4)."""
    from src.data import external_sources as xs
    keys = []
    if (xs.EXTERNAL / "hongkong/dataset.zip").exists():
        catalogue = xs.hongkong_catalogue()
        keys += [f"hongkong:{name}" for name in catalogue[catalogue.file.notna()].name]
    if UTRECHT_HOURLY.exists():
        keys += [f"utrecht:{i}" for i in xs.utrecht_catalogue().ID]
    keys += fmi_candidates("R")
    return keys


UTRECHT_HOURLY = ROOT / "data/raw/external/utrecht/hourly_complete.parquet"

# Finnish Meteorological Institute outdoor solar laboratories (Karhu et al. 2026, Geoscience Data Journal 13, e70039;
# data doi:10.57707/fmi-b2share.tn0wb-as670, CC BY 4.0). 1-min data, time stamp = end of the window, UTC.
# Helsinki and Kuopio have an on-site plane-of-array pyranometer (15 deg): Tier S. Sodankyla 20 deg has only GHI: Tier R.
# No nameplate capacity is published in machine-readable form: dc_kw is estimated from the record (see build_fmi_hourly)
# and only fixes the normalisation; the azimuth (south) is an assumption that matters only for the clear-sky diagnostic.
FMI_DIR = ROOT / "data/raw/external/fmi_finland"
FMI_HOURLY = FMI_DIR / "hourly_complete.parquet"
FMI_SITES = {
    "Helsinki": {"file": "FMI_Helsinki_PV.csv", "tier": "S", "lat": 60.2046, "lon": 24.9621, "tilt": 15.0,
                 "poa": "GLOBA_PT1M_AVG(:31)", "tamb": "TA_PT1M_AVG(:31)", "tmod": ["TTECH_PT1M_AVG(:32)", "TTECH_PT1M_AVG(:33)"],
                 "ghi": "GLOB_PT1M_AVG", "power": "pv_inv_out", "dc": "pv_inv_in", "location": "Helsinki, Finland"},
    "Kuopio": {"file": "FMI_Kuopio_PV.csv", "tier": "S", "lat": 62.8917, "lon": 27.6317, "tilt": 15.0,
               "poa": "GLOBA_PT1M_AVG(:31)", "tamb": "TA_PT1M_AVG(:31)", "tmod": ["TTECH_PT1M_AVG(:32)", "TTECH_PT1M_AVG(:33)"],
               "ghi": "GLOB_PT1M_AVG", "power": "pv_inv_out", "dc": "pv_inv_in", "location": "Kuopio, Finland"},
    "Sodankyla20": {"file": "FMI_Sodankyla_20deg_PV.csv", "tier": "R", "lat": 67.3668, "lon": 26.6289, "tilt": 20.0,
                    "poa": None, "tamb": "TA_PT1M_AVG(:101)", "tmod": ["TTECH_PT1M_AVG(:102)"],
                    "ghi": "GLOB_PT1M_AVG", "power": "pv_inv_out", "dc": None, "location": "Sodankyla, Finland"},
}
FMI_SNOW_MAX = 1        # hours with visual snow status >= 2 (probably covered) are dropped


def fmi_candidates(tier: str) -> list[str]:
    return [f"fmi:{k}" for k, v in FMI_SITES.items() if v["tier"] == tier and (FMI_DIR / v["file"]).exists()]


def build_fmi_hourly(chunk_rows: int = 500_000) -> Path:
    """FMI 1-min CSVs -> hourly means of complete hours (>= 54 of 60 power samples), snow hours removed, per site.
    The window label is the end of the minute, so a stamp belongs to the hour of (stamp - 30 s)."""
    parts = {}
    for site, spec in FMI_SITES.items():
        path = FMI_DIR / spec["file"]
        with open(path, encoding="utf-8", errors="replace") as handle:
            skip = 0
            for line in handle:
                if line.startswith("#"):
                    skip += 1
                else:
                    break
        want = ["utctime", spec["power"], "vis_SnoP", spec["ghi"], spec["tamb"]] + spec["tmod"]
        want += [c for c in (spec["poa"], spec["dc"]) if c]
        sums, counts = [], []
        for chunk in pd.read_csv(path, sep=";", skiprows=skip, usecols=want, chunksize=chunk_rows, low_memory=False):
            stamp = pd.to_datetime(chunk.pop("utctime")) - pd.Timedelta(seconds=30)
            values = chunk.apply(pd.to_numeric, errors="coerce").astype("float64")
            values["snow_max"] = values["vis_SnoP"]
            values.index = stamp.dt.floor("h").to_numpy()
            sums.append(values.drop(columns=["snow_max"]).groupby(level=0).sum(min_count=1))
            counts.append(values.drop(columns=["snow_max"]).notna().groupby(level=0).sum())
            sums.append(values[["snow_max"]].groupby(level=0).max().rename(columns={"snow_max": "snow_max"}))
            counts.append(None)
        snow = pd.concat([s for s in sums if list(s.columns) == ["snow_max"]]).groupby(level=0).max()["snow_max"]
        total = pd.concat([s for s in sums if list(s.columns) != ["snow_max"]]).groupby(level=0).sum(min_count=1)
        n = pd.concat([c for c in counts if c is not None]).groupby(level=0).sum()
        hourly_mean = total / n.replace(0, np.nan)
        out = pd.DataFrame({"power_w": hourly_mean[spec["power"]].where(n[spec["power"]] >= 54),
                            "ghi_wm2": hourly_mean[spec["ghi"]].where(n[spec["ghi"]] >= 54),
                            "t_amb_c": hourly_mean[spec["tamb"]].where(n[spec["tamb"]] >= 54)})
        if spec["poa"]:
            out["poa_wm2"] = hourly_mean[spec["poa"]].where(n[spec["poa"]] >= 54)
        tm = pd.concat([hourly_mean[c].where(n[c] >= 54) for c in spec["tmod"]], axis=1).mean(axis=1)
        out["t_module_c"] = tm
        # The inverters (Helsinki, Kuopio) and micro-inverters (Sodankyla) log nothing at night: hours with a measured
        # irradiance below 5 W/m2 and no power record are hours of zero output (complete grid for the 24-h windows).
        irr_all = out.poa_wm2 if spec["poa"] else out.ghi_wm2
        grid = pd.date_range(out.index.min(), out.index.max(), freq="h")
        out = out.reindex(grid)
        irr_all = (out.poa_wm2 if spec["poa"] else out.ghi_wm2)
        out.loc[out.power_w.isna() & irr_all.notna() & (irr_all < 5), "power_w"] = 0.0
        out = out[out.power_w.notna()]
        out.loc[snow.reindex(out.index).fillna(0) > FMI_SNOW_MAX, ["power_w", "poa_wm2", "ghi_wm2", "t_module_c"] if spec["poa"] else ["power_w", "ghi_wm2", "t_module_c"]] = np.nan
        out = out[out.power_w.notna()]
        ref_col = hourly_mean[spec["dc"]] if spec["dc"] else hourly_mean[spec["power"]]
        irr = out.poa_wm2 if spec["poa"] else out.ghi_wm2
        ref = ref_col.reindex(out.index)[irr.reindex(out.index) > 700]
        out.attrs["dc_kw_estimate"] = float(ref.quantile(0.99) / 1000.0) if len(ref) > 50 else float(out.power_w.quantile(0.999) / 1000.0)
        out["site"] = site
        out["dc_kw_estimate"] = out.attrs["dc_kw_estimate"]
        parts[site] = out
    allsites = pd.concat(parts.values())
    allsites.index.name = "timestamp_raw"
    allsites.astype({c: "float32" for c in allsites.columns if c != "site"}).to_parquet(FMI_HOURLY)
    return FMI_HOURLY


def build_utrecht_hourly(chunk_rows: int = 200_000) -> Path:
    """Unfiltered 1-min Utrecht file (W, UTC) -> hourly means of complete hours (>= 54 of 60 samples), per system."""
    source = ROOT / "data/raw/external/utrecht/unfiltered_pv_power_measurements.csv"
    sums, counts = [], []
    for chunk in pd.read_csv(source, chunksize=chunk_rows):
        stamps = pd.to_datetime(chunk.pop("DateTime"), utc=True).dt.tz_localize(None).dt.floor("h")
        values = chunk.apply(pd.to_numeric, errors="coerce").astype("float64")
        values.index = stamps.to_numpy()
        sums.append(values.groupby(level=0).sum(min_count=1))
        counts.append(values.notna().groupby(level=0).sum())
    total = pd.concat(sums).groupby(level=0).sum(min_count=1)
    n = pd.concat(counts).groupby(level=0).sum()
    hourly = (total / n.replace(0, np.nan)).where(n >= 54)
    hourly.index.name = "timestamp_raw"
    hourly.astype("float32").to_parquet(UTRECHT_HOURLY)
    return UTRECHT_HOURLY


def system_key(sid: str) -> str:
    """Result-table id of a candidate key."""
    if sid.startswith("utrecht:"):
        return f"ut_{sid.split(':', 1)[1]}"
    if sid.startswith("fmi:"):
        return f"fmi_{sid.split(':', 1)[1].lower()}"
    if sid.startswith("hongkong:"):
        from src.data.external_sources import _norm
        return f"hk_{_norm(sid.split(':', 1)[1])[:24]}"
    return f"pvdaq_{sid}"


def location_name(text: str) -> str:
    """Catalogue site_location, normalised ('Golden CO' -> 'Golden, CO'): the independent unit."""
    text = str(text).strip()
    return text if "," in text else re.sub(r"\s+([A-Z]{2})$", r", \1", text)


def catalogue_row(sid: str) -> dict:
    cat = pd.read_csv(CATALOGUE).drop_duplicates("system_id")
    row = cat[cat.system_id == int(sid)]
    if row.empty:
        raise KeyError(f"system {sid} not in the PVDAQ catalogue")
    row = row.iloc[0]
    tz = TZ.get(sid, str(row.timezone_or_utc_offset))
    tz = TZ_CODES.get(tz, tz)
    tilt, azimuth = PRIZE_ORIENTATION.get(sid, (row.tilt, row.azimuth))
    return {"lat": float(row.latitude), "lon": float(row.longitude), "dc_kw": float(row.dc_capacity_kW),
            "tz": tz, "tilt": float(tilt) if pd.notna(tilt) else np.nan,
            "azimuth": float(azimuth) if pd.notna(azimuth) and float(azimuth) >= 0 else np.nan,
            "tracking": str(row.tracking) == "tracking", "location": location_name(row.site_location),
            "name": str(row.system_public_name), "catalogue_climate": str(row.kg_climate)}


# ------------------------------------------------------------------------------------------ external sources
def climate_class(lat: float, lon: float) -> str:
    """Köppen-Geiger 1991-2020 class from the verified 1-km raster (rule 5); 'pending: ...' when unavailable."""
    try:
        from src.climate.classify import classify_sites, verify_reference
        raster, codebook, _ = verify_reference(ROOT / "data/reference/climate/source.json")
        return str(classify_sites(pd.DataFrame([{"system_id": "x", "latitude": lat, "longitude": lon}]),
                                  raster, codebook).climate_class.iloc[0])
    except Exception as exc:  # recorded, never replaced by the catalogue label
        return f"pending: raster unavailable ({type(exc).__name__})"


ERA5_VARIABLES = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m")


def era5(meta: dict, start: str, end: str, variables: tuple[str, ...]) -> pd.DataFrame:
    """ERA5 at the site. One request per site and span with every variable the expansion uses (Open-Meteo counts
    requests by span, not by variables up to ten), so later calls for another subset hit the cache.
    ``era5_lat``/``era5_lon``/``era5_span`` in meta (Hong Kong, Utrecht) give co-located systems one shared request
    (nearest 0.25-degree cell, the dataset's whole span); callers align by time stamp, so the result is unchanged."""
    from src.weather.open_meteo import reanalysis
    lat, lon = meta.get("era5_lat", meta["lat"]), meta.get("era5_lon", meta["lon"])
    start, end = meta.get("era5_span", (start, end))
    frame, _ = reanalysis(lat, lon, start, end, ROOT / "data/cache/weather", variables=ERA5_VARIABLES)
    return frame[list(variables)]


def via_shared_era5(fn, frame: pd.DataFrame, meta: dict, *args, **kwargs):
    """Run a tier_a function (convention, drift) unchanged, but serve its ERA5 from the system's shared request."""
    original, months = tier_a.reanalysis, tier_a.MIN_DRIFT_MONTHS
    tier_a.reanalysis = lambda lat, lon, start, end, cache, variables: (era5(meta, *span(frame.index), variables), {})
    if str(meta.get("system_id", "")).startswith(("hk_", "ut_", "fmi_")):       # 2-3 y records: 24 matched months
        tier_a.MIN_DRIFT_MONTHS = 24
    try:
        return fn(frame, meta, *args, **kwargs)
    finally:
        tier_a.reanalysis, tier_a.MIN_DRIFT_MONTHS = original, months


def span(index: pd.DatetimeIndex) -> tuple[str, str]:
    """Padded UTC-date span of a local-time index (the same for every ERA5 call of a system)."""
    return str((index.min() - pd.Timedelta(days=1)).date()), str((index.max() + pd.Timedelta(days=1)).date())


def era5_poa(meta: dict, index: pd.DatetimeIndex) -> pd.Series:
    """ERA5 plane-of-array irradiance on local-standard-time interval-start labels (Tier R)."""
    if meta.get("irradiance_override") == "era5_ghi":        # horizontal approximation, no published orientation
        ghi = era5(meta, *span(index), ("shortwave_radiation",)).shortwave_radiation
        return tier_a.era5_local(ghi, meta["std_offset_h"], index)
    weather = era5(meta, *span(index), ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"))
    return tier_a.era5_local(transpose(weather, meta), meta["std_offset_h"], index)


def transpose(weather: pd.DataFrame, meta: dict) -> pd.Series:
    """GHI/DNI/DHI (UTC interval-start labels) -> POA with the published tilt/azimuth, sun at the midpoint."""
    import pvlib
    mid = weather.index + pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(meta["lat"], meta["lon"])
    sun = loc.get_solarposition(mid)
    dni_extra = pvlib.irradiance.get_extra_radiation(mid)
    airmass = pvlib.atmosphere.get_relative_airmass(sun.apparent_zenith.to_numpy())
    poa = pvlib.irradiance.get_total_irradiance(
        meta["tilt"], meta["azimuth"], sun.apparent_zenith.to_numpy(), sun.azimuth.to_numpy(),
        weather.direct_normal_irradiance.to_numpy(), weather.shortwave_radiation.to_numpy(),
        weather.diffuse_radiation.to_numpy(), dni_extra=dni_extra.to_numpy(), airmass=airmass,
        albedo=TRANSPOSITION["albedo"], model=TRANSPOSITION["model"])
    return pd.Series(np.nan_to_num(np.asarray(poa["poa_global"], float), nan=0.0).clip(min=0), index=weather.index)


def clearsky_poa(meta: dict, labels_utc: pd.DatetimeIndex) -> np.ndarray:
    """pvlib Ineichen clear-sky irradiance on the array plane (horizontal when tilt is unknown)."""
    import pvlib
    mid = labels_utc + pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(meta["lat"], meta["lon"])
    cs = loc.get_clearsky(mid, model="ineichen")
    if not np.isfinite(meta.get("tilt", np.nan)) or not np.isfinite(meta.get("azimuth", np.nan)):
        return cs.ghi.to_numpy()
    sun = loc.get_solarposition(mid)
    poa = pvlib.irradiance.get_total_irradiance(meta["tilt"], meta["azimuth"], sun.apparent_zenith, sun.azimuth,
                                                cs.dni, cs.ghi, cs.dhi, model="isotropic")
    return poa["poa_global"].fillna(0).to_numpy()


# ------------------------------------------------------------------------------------------------ loading
def native_file(sid: str) -> Path:
    pattern = f"prize_{sid}_last*m.parquet" if sid in PRIZE else f"pvdaq_{sid}_*.parquet"
    files = sorted(RAW.glob(pattern), key=lambda p: p.stat().st_size)
    if not files:
        raise FileNotFoundError(f"{sid}: no native file under {RAW} (run scripts/fetch_climate_expansion.py)")
    return files[-1]


def hourly(sid: str) -> tuple[pd.DataFrame, dict]:
    """Hourly frame in the raw clock (complete hours only, as tier_a._hourly) plus metadata."""
    if sid.startswith("fmi:"):
        return hourly_fmi(sid)
    if sid.startswith("hongkong:"):
        return hourly_hongkong(sid)
    if sid.startswith("utrecht:"):
        return hourly_utrecht(sid)
    native, meter_info = meter_handling(pd.read_parquet(native_file(sid)))
    out, step = _aggregate(native)
    meter_info["era5_span"] = span(out.index)     # every ERA5 use of the system shares this request
    meta = {"system_id": system_key(sid), "raw_id": sid, "tier": tier_of(sid), "source": "PVDAQ",
            **catalogue_row(sid), "native_interval_minutes": step, **meter_info}
    meta["label"] = f"{meta['name']}, {meta['location']}"
    meta["std_offset_h"] = tier_a.std_offset_h(meta["tz"])
    info = native_file(sid).with_suffix(".json")
    meta["time_basis"] = json.loads(info.read_text()).get("time_basis", "measured_on") if info.exists() else "measured_on"
    return out, meta


def hourly_hongkong(sid: str) -> tuple[pd.DataFrame, dict]:
    """Hong Kong station (Dryad doi:10.5061/dryad.m37pvmd99) with the territory's on-site GHI and air temperature."""
    from src.data import external_sources as xs
    name = sid.split(":", 1)[1]
    native, site = xs.hongkong(name)
    out, step = _aggregate(native)
    out = out.drop(columns=[c for c in ("ghi_wm2", "poa_wm2") if c in out])      # sensor step in 2023-02 (plan, 2026-10-01)
    meta = {"system_id": system_key(sid), "raw_id": sid, "tier": "R", "source": "Hong Kong (Dryad doi:10.5061/dryad.m37pvmd99)",
            "irradiance_override": "era5_ghi",
            "lat": float(site["latitude"]), "lon": float(site["longitude"]), "dc_kw": float(site["dc_capacity_kw"]),
            "tz": site["tz_name"], "tilt": np.nan, "azimuth": np.nan, "tracking": False, "location": "Hong Kong",
            "name": name, "catalogue_climate": "", "native_interval_minutes": step, "time_basis": "unknown"}
    meta["label"] = f"{name}, Hong Kong"
    meta.update(era5_lat=round(meta["lat"] * 4) / 4, era5_lon=round(meta["lon"] * 4) / 4,
                era5_span=("2020-12-31", "2024-01-01"))
    meta["std_offset_h"] = tier_a.std_offset_h(meta["tz"])
    return out, meta


def hourly_fmi(sid: str) -> tuple[pd.DataFrame, dict]:
    """FMI outdoor solar laboratory (Finland): hourly complete-hour means from build_fmi_hourly (raw clock = UTC)."""
    name = sid.split(":", 1)[1]
    spec = FMI_SITES[name]
    if not FMI_HOURLY.exists():
        build_fmi_hourly()
    table = pd.read_parquet(FMI_HOURLY)
    table = table[table.site == name]
    dc_kw = float(table.dc_kw_estimate.iloc[0])
    frame = table.drop(columns=["site", "dc_kw_estimate"]).astype("float64")
    frame = frame.dropna(axis=1, how="all")
    meta = {"system_id": system_key(sid), "raw_id": sid, "tier": spec["tier"], "source": "FMI outdoor solar laboratories (CC BY 4.0)",
            "lat": spec["lat"], "lon": spec["lon"], "dc_kw": dc_kw, "tz": "Europe/Helsinki", "tilt": spec["tilt"],
            "azimuth": 180.0, "tracking": False, "location": spec["location"], "name": f"FMI {name}",
            "catalogue_climate": "", "native_interval_minutes": 1, "time_basis": "UTC (published)",
            "dc_capacity_basis": "estimated from the record (p99 of DC input at POA > 700 W/m2; no machine-readable nameplate)"}
    meta["label"] = f"FMI {name}, {spec['location']}"
    meta["std_offset_h"] = tier_a.std_offset_h(meta["tz"])
    meta.update(era5_lat=round(spec["lat"] * 4) / 4, era5_lon=round(spec["lon"] * 4) / 4,
                era5_span=span(frame.index))
    return frame, meta


def hourly_utrecht(sid: str) -> tuple[pd.DataFrame, dict]:
    """Utrecht system (Zenodo 10953360): hourly complete-hour means built by build_utrecht_hourly (raw clock = UTC)."""
    from src.data import external_sources as xs
    name = sid.split(":", 1)[1]
    power = pd.read_parquet(UTRECHT_HOURLY, columns=[name])[name].astype("float64").dropna()
    catalogue = xs.utrecht_catalogue()
    site = xs.utrecht_site(catalogue[catalogue.ID == name].iloc[0])
    meta = {"system_id": system_key(sid), "raw_id": sid, "tier": "R", "source": "Utrecht (Zenodo 10953360)",
            "lat": float(site["latitude"]), "lon": float(site["longitude"]), "dc_kw": float(site["dc_capacity_kw"]),
            "tz": site["tz_name"], "tilt": float(site["tilt"]), "azimuth": float(site["azimuth"]), "tracking": False,
            "location": "Utrecht", "name": name, "catalogue_climate": "", "native_interval_minutes": 1,
            "time_basis": "UTC (published)"}
    meta["label"] = f"{name}, Utrecht"
    meta["std_offset_h"] = tier_a.std_offset_h(meta["tz"])
    if np.isfinite(meta["lat"]) and np.isfinite(meta["lon"]):
        meta.update(era5_lat=round(meta["lat"] * 4) / 4, era5_lon=round(meta["lon"] * 4) / 4,
                    era5_span=("2013-12-31", "2018-01-01"))
    return pd.DataFrame({"power_w": power}), meta


def _aggregate(native: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rename = {"onsite_poa_wm2": "poa_wm2", "onsite_ambient_temp_c": "t_amb_c", "onsite_module_temp_c": "t_module_c",
              "onsite_ghi_wm2": "ghi_wm2"}
    frame = native.rename(columns=rename)
    keep = [c for c in ("power_w", "poa_wm2", "ghi_wm2", "t_amb_c", "t_module_c") if c in frame]
    frame = frame[keep]
    hours = frame.index.floor("h")
    out = frame.groupby(hours).mean()
    step = max(1, int(round(frame.index.to_series().diff().dt.total_seconds().median() / 60)))
    grp = frame.power_w.groupby(hours)                 # PV_HOURLY_RULE: see tier_a._hourly
    counts = grp.count() if os.environ.get("PV_HOURLY_RULE", "valid") == "valid" else grp.size()
    return out[counts >= int(np.ceil(0.9 * 60 / step))], step


def _convention_scores(signal: pd.Series, reference_at, meta: dict, threshold: float = 20.0) -> dict:
    """Hypotheses as tier_a.detect_time_convention; ``reference_at(labels_utc)`` gives the reference."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(meta["tz"])
    dst = zone.utcoffset(datetime(2021, 1, 15)) != zone.utcoffset(datetime(2021, 7, 15))
    scores = {}
    for conv in (("local_standard", "local_dst", "utc") if dst else ("local_standard", "utc")):
        labels = tier_a.to_utc(signal.index, meta["tz"], conv)
        both = pd.DataFrame({"s": signal.to_numpy(), "e": reference_at(labels)}, index=labels)
        both = both[both.index.notna()].dropna()
        both = both[(both.s > threshold) | (both.e > 20)]
        local = both.index.tz_convert(meta["tz"])
        in_dst = np.array([bool(t.dst()) for t in local]) if dst else np.zeros(len(both), bool)
        scores[conv] = {"r_all": float(both.s.corr(both.e)),
                        "r_dst_season": float(both[in_dst].s.corr(both[in_dst].e)) if in_dst.sum() > 500 else np.nan}
    rank = sorted(scores, key=lambda c: -scores[c]["r_all"])
    best, second = rank[0], rank[1]
    key = "r_dst_season" if {best, second} == {"local_standard", "local_dst"} else "r_all"
    if key == "r_dst_season" and scores[second][key] > scores[best][key]:
        best, second = second, best
    return {"convention": best, "margin": round(scores[best][key] - scores[second][key], 4),
            "r": round(scores[best]["r_all"], 4)}


def detect_convention(frame: pd.DataFrame, meta: dict) -> dict:
    """Final clock convention against ERA5 (rule 1) and a clear-sky diagnostic (provisional only).
    Tier S: on-site POA vs ERA5 GHI (tier_a.detect_time_convention). Tier R: AC power vs ERA5 POA,
    or the on-site GHI vs ERA5 GHI where a GHI sensor exists."""
    out = {}
    threshold = 20.0
    if has(frame, "poa_wm2"):
        signal, plane = frame.poa_wm2, False
    elif has(frame, "ghi_wm2"):
        signal, plane = frame.ghi_wm2, False
    else:   # power as the signal: 2 % of its p99 plays the role of 20 W/m2
        signal, plane = frame.power_w.clip(lower=0), True
        threshold = 0.02 * float(signal.quantile(0.99))
    cs_meta = meta if plane else {**meta, "tilt": np.nan}
    diag = _convention_scores(signal, lambda labels: _nan_safe(clearsky_poa, cs_meta, labels), meta, threshold)
    out.update({"convention_clearsky": diag["convention"], "convention_clearsky_margin": diag["margin"]})
    try:
        if meta["tier"] == "S":           # rule 1 exactly as in the Tier-A screen
            final = via_shared_era5(tier_a.detect_time_convention, frame, meta)
            out.update({**final, "era5": "ok"})
            return out
        start, end = span(frame.index)
        if plane and meta.get("irradiance_override") == "era5_ghi":
            ref = era5(meta, start, end, ("shortwave_radiation",)).shortwave_radiation
        elif plane:
            weather = era5(meta, start, end, ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"))
            ref = transpose(weather, meta)
        else:
            ref = era5(meta, start, end, ("shortwave_radiation",)).shortwave_radiation
        final = _convention_scores(signal, lambda labels: tier_a.era5_at(ref, labels), meta, threshold)
        out.update({"time_convention": final["convention"], "convention_margin": final["margin"],
                    "convention_r": final["r"], "era5": "ok"})
    except Exception as exc:
        out.update({"time_convention": "pending", "era5": f"unavailable: {type(exc).__name__}: {str(exc)[:120]}"})
    return out


def has(frame: pd.DataFrame, column: str) -> bool:
    return column in frame and frame[column].notna().mean() >= 0.3


def _nan_safe(fn, meta, labels):
    values = np.full(len(labels), np.nan)
    ok = ~pd.isna(labels)
    values[ok] = fn(meta, pd.DatetimeIndex(labels[ok]))
    return values


def restamp(frame: pd.DataFrame, meta: dict, convention: str) -> pd.DataFrame:
    """Raw clock -> local standard time (tier_a.load_system)."""
    utc = tier_a.to_utc(frame.index, meta["tz"], convention)
    out = frame[utc.notna()].copy()
    out.index = (utc[utc.notna()] + pd.Timedelta(hours=meta["std_offset_h"])).tz_localize(None)
    return out[~out.index.duplicated()].sort_index()


def load(sid: str, era5_temperature: bool = True) -> tuple[pd.DataFrame, dict]:
    """Analysis frame (local standard time) for the rate pipeline; raises when ERA5 checks are pending."""
    frame, meta = hourly(sid)
    meta.update(detect_convention(frame, meta))
    meta["climate"] = climate_class(meta["lat"], meta["lon"])
    if meta["time_convention"] == "pending":
        raise RuntimeError(f"{sid}: clock convention needs ERA5 ({meta['era5']})")
    frame = restamp(frame, meta, meta["time_convention"])
    if meta["tier"] == "R" and has(frame, "ghi_wm2"):
        frame["poa_wm2"] = frame.ghi_wm2
        meta["irradiance_basis"] = "on-site GHI (horizontal approximation)"
    elif meta["tier"] == "R":
        frame["poa_wm2"] = era5_poa(meta, frame.index)
        meta["irradiance_basis"] = ("ERA5 GHI (horizontal approximation)" if meta.get("irradiance_override")
                                    else f"ERA5 transposed ({TRANSPOSITION['model']}, tilt {meta['tilt']}, az {meta['azimuth']})")
    else:
        meta["irradiance_basis"] = "on-site POA"
    meta["temperature_basis"] = "on-site"
    if "t_amb_c" not in frame and "t_module_c" not in frame and era5_temperature:
        t = era5(meta, *span(frame.index), ("temperature_2m",))
        frame["t_amb_c"] = tier_a.era5_local(t.temperature_2m, meta["std_offset_h"], frame.index)
        meta["temperature_basis"] = "ERA5 2-m temperature"
    return frame.drop(columns=[c for c in ("ghi_wm2",) if c in frame]), meta


# ------------------------------------------------------------------------------------------------ screens
def _valid_years(power: pd.Series, irradiance: pd.Series) -> float:
    day = pd.DataFrame({"p": power, "g": irradiance})
    day = day[(day.g > 50) & day.p.notna()]
    return float((day.groupby(day.index.date).size() >= 8).sum() / 365.25)


def meter_handling(native: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """9069 has two revenue meters. Registered rule: midday median meter2/meter1 in 0.95-1.05 -> redundant,
    meter 1 alone; otherwise the sum. Amendment (2026-09-30, before any rate): meters that never log at the same
    time (< 1 % of either meter's samples overlap) are successive meters -- a meter replacement -- and are
    spliced (meter 1, then meter 2); the change date is recorded for a step check in the rate analysis."""
    if "meter2_power_w" not in native:
        return native, {}
    m1, m2 = native.power_w, native.meter2_power_w
    both = m1.notna() & m2.notna()
    info = {"meter1_samples": int(m1.notna().sum()), "meter2_samples": int(m2.notna().sum()),
            "meter_overlap_samples": int(both.sum())}
    if both.sum() < 0.01 * min(m1.notna().sum(), m2.notna().sum()):
        info.update({"power_channel": "meter 1 then meter 2 (successive, spliced)",
                     "meter1_last": str(m1.dropna().index.max()), "meter2_first": str(m2.dropna().index.min())})
        power = m1.combine_first(m2)
    else:
        mid = native[both & (native.index.hour >= 11) & (native.index.hour <= 13) & (m1 > 0) & (m2 > 0)]
        ratio = float((mid.meter2_power_w / mid.power_w).median())
        redundant = bool(0.95 <= ratio <= 1.05)
        info.update({"meter2_over_meter1": round(ratio, 3),
                     "power_channel": "meter 1 (meters redundant)" if redundant else "meter 1 + meter 2"})
        power = m1 if redundant else m1 + m2
    out = native.drop(columns=["meter2_power_w"]).assign(power_w=power)
    return out[out.power_w.notna()], info


def screen_system(sid: str) -> dict:
    """Screen one candidate; returns a flat row with a decision in {pass, fail, pending}."""
    try:
        frame, meta = hourly(sid)
    except FileNotFoundError as exc:
        return {"system_id": system_key(sid), "raw_id": sid, "tier": tier_of(sid), "decision": "pending",
                "reason": str(exc)}
    row = {"system_id": meta["system_id"], "raw_id": sid, "tier": meta["tier"], "location": meta["location"], "name": meta["name"],
           "lat": meta["lat"], "lon": meta["lon"], "dc_kw": meta["dc_kw"], "tilt": meta["tilt"],
           "azimuth": meta["azimuth"], "tracking": meta["tracking"], "catalogue_climate": meta["catalogue_climate"],
           "climate": climate_class(meta["lat"], meta["lon"]) if np.isfinite(meta["lat"]) else "unknown",
           "time_basis": meta["time_basis"],
           "native_interval_min": meta["native_interval_minutes"],
           "first": str(frame.index.min().date()), "last": str(frame.index.max().date()),
           "archived_years": round((frame.index.max() - frame.index.min()).days / 365.25, 2)}
    row.update({k: v for k, v in meta.items() if k.startswith("meter") or k == "power_channel"})
    missing = [what for what, ok in (("coordinates", np.isfinite(meta["lat"]) and np.isfinite(meta["lon"])),
                                     ("DC capacity", np.isfinite(meta["dc_kw"]) and meta["dc_kw"] > 0)) if not ok]
    if missing:     # climate class, ERA5 and the scale/night rules all need them
        return {**row, "decision": "fail", "reason": f"no {' and '.join(missing)} in the metadata"}
    row.update(detect_convention(frame, meta))
    final = row["time_convention"] != "pending"
    frame_std = restamp(frame, meta, row["time_convention"] if final else row["convention_clearsky"])
    meta["climate"] = row["climate"]
    return {**row, **(screen_s(frame_std, meta, final) if meta["tier"] == "S" else screen_r(frame_std, meta, final))}


def screen_s(frame: pd.DataFrame, meta: dict, final_clock: bool) -> dict:
    """tier_a.screen unchanged (scale, night, sensor, >= 4 y, two-reference drift)."""
    if not has(frame, "poa_wm2"):
        return {"decision": "fail", "reason": "no usable on-site POA"}
    res = via_shared_era5(tier_a.screen, frame, meta)
    reasons = [r for r in res.get("reason", "").split("; ") if r]
    short = str(meta["system_id"]).startswith("fmi_")       # amendment 2026-10-01: Finnish records, 2 valid years
    if short and res.get("valid_years", 0) >= MIN_YEARS_R_SHORT:
        reasons = [r for r in reasons if not r.startswith("< 4")]
    pending = []
    if not final_clock:
        pending.append("clock convention vs ERA5")
    if "poa_era5_ratio_trend_per_year" not in res:
        pending.append("POA drift vs ERA5")
    if str(meta["climate"]).startswith("pending"):
        pending.append("Köppen raster")
    out = {k: v for k, v in res.items() if k not in ("system_id", "climate", "label", "first", "last", "pass", "reason")}
    out["tier_label"] = "S-short" if short else "S"
    decision = "fail" if reasons else ("pending" if pending else "pass")
    return {**out, "decision": decision, "reason": "; ".join(reasons), "pending": "; ".join(pending)}


def screen_r(frame: pd.DataFrame, meta: dict, final_clock: bool) -> dict:
    """Tier R: scale (final); night offset and >= 3 y of valid days on the irradiance input (ERA5 POA, or the
    on-site GHI where it exists; clear-sky diagnostics while ERA5 is pending); two-reference drift on on-site GHI."""
    cap = meta["dc_kw"] * 1000.0
    reasons, pending = [], []
    ghi = has(frame, "ghi_wm2")
    out = {"irradiance_input": "on-site GHI" if ghi else ("ERA5 GHI" if meta.get("irradiance_override") else "ERA5 POA"),
           "p999_over_dc": round(float(frame.power_w.quantile(0.999)) / cap, 3)}
    if out["p999_over_dc"] > 1.10:
        reasons.append("power above 1.10 x DC rating")
    labels = (frame.index - pd.Timedelta(hours=meta["std_offset_h"])).tz_localize("UTC")
    cs = pd.Series(clearsky_poa(meta if not ghi else {**meta, "tilt": np.nan}, labels), index=frame.index)
    night_cs = frame.power_w[cs < 5].abs()
    out["night95_over_dc_clearsky_diag"] = round(float(night_cs.quantile(0.95)) / cap, 4) if len(night_cs) else np.nan
    out["valid_years_clearsky_upper_diag"] = round(_valid_years(frame.power_w, cs), 2)
    if (not ghi and meta.get("irradiance_override") != "era5_ghi"
            and (not np.isfinite(meta.get("tilt", np.nan)) or not np.isfinite(meta.get("azimuth", np.nan)))):
        reasons.append("no published tilt/azimuth for the ERA5 transposition")
    try:
        if ghi:
            irr = frame.ghi_wm2
            if not final_clock:     # the GHI rules do not need ERA5, but the day boundaries need the clock
                raise RuntimeError("clock convention pending")
        else:
            if not final_clock:
                raise RuntimeError("clock convention pending")
            irr = era5_poa(meta, frame.index)
        night = frame.power_w[irr < 5].abs()
        out["night95_over_dc"] = round(float(night.quantile(0.95)) / cap, 4) if len(night) else 0.0
        out["valid_years"] = round(_valid_years(frame.power_w, irr), 2)
        if out["night95_over_dc"] > 0.01:
            reasons.append("night power above 1 % of rating")
        short = str(meta["system_id"]).startswith(("hk_", "ut_", "fmi_"))
        out["tier_label"] = "R-short" if short else "R"
        min_years = MIN_YEARS_R_SHORT if short else MIN_YEARS["R"]
        if out["valid_years"] < min_years:
            reasons.append(f"< {min_years:g} years of valid days")
    except Exception as exc:
        pending.append(f"night offset and valid years ({type(exc).__name__}: {str(exc)[:60]})")
    if ghi:
        g = frame.drop(columns=[c for c in ("poa_wm2",) if c in frame]).assign(poa_wm2=frame.ghi_wm2)
        try:
            out.update(via_shared_era5(tier_a.sensor_drift, g, meta))
        except Exception as exc:
            pending.append(f"GHI drift vs ERA5 ({type(exc).__name__})")
        try:
            out.update(tier_a.clearsky_drift(g, meta))
        except Exception as exc:
            out["clearsky_check"] = f"not computed: {type(exc).__name__}: {exc}"
        era, csd = out.get("poa_era5_ratio_trend_per_year", np.nan), out.get("poa_clearsky_envelope_trend_per_year", np.nan)
        if np.isfinite(era) and np.isfinite(csd) and abs(era) > 0.01 and abs(csd) > 0.01 and np.sign(era) == np.sign(csd):
            reasons.append("GHI sensor drift (> 1 %/yr vs ERA5 and vs clear-sky envelope)")
    if not final_clock:
        pending.append("clock convention vs ERA5")
    if str(meta["climate"]).startswith("pending"):
        pending.append("Köppen raster")
    decision = "fail" if reasons else ("pending" if pending else "pass")
    return {**out, "decision": decision, "reason": "; ".join(reasons), "pending": "; ".join(pending)}


def select_main(screen: pd.DataFrame, existing: dict[str, int]) -> pd.DataFrame:
    """Rule 4: at most 3 systems per location in the main statistics (counting systems already in the paper's
    main set), chosen by record length (valid years, then archived span) before any rate. Pending systems get a
    provisional rank that is recomputed once their screen is final."""
    rows = []
    for tier in ("S", "R"):
        part = screen[(screen.tier == tier) & screen.decision.isin(["pass", "pending"])].copy()
        if part.empty:
            continue
        length = part.valid_years if "valid_years" in part else pd.Series(np.nan, index=part.index)
        part["rank_key"] = length.fillna(-1)
        for location, group in part.groupby("location"):
            group = group.sort_values(["rank_key", "archived_years"], ascending=False)
            slots = max(0, MAX_PER_LOCATION - existing.get(location, 0))
            for i, (_, r) in enumerate(group.iterrows()):
                role = "main" if i < slots else "sensitivity"
                status = "final" if r.decision == "pass" and not group.decision.eq("pending").any() else "provisional"
                raw = r.get("raw_id")
                rows.append({"system_id": r.system_id,
                             "raw_id": raw if isinstance(raw, str) else str(r.system_id).replace("pvdaq_", ""),
                             "tier": tier, "location": location, "climate": r.climate,
                             "rank_in_location": i + 1, "existing_main_at_location": existing.get(location, 0),
                             "role": role, "status": status})
    return pd.DataFrame(rows)

