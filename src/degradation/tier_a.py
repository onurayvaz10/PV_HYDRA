"""Tier-A systems for the degradation paper: one loader, one screen.

Every system becomes an hourly frame (interval-start labels, local standard time; the raw
clock convention is detected against ERA5, see detect_time_convention)
with power_w, poa_wm2, t_amb_c, t_module_c (when measured) plus metadata. The
screen is specified in the project code and model-independent; independent
preregistration has not been established:
  scale    p99.9(P) <= 1.10 x DC rating         (unit/meter errors)
  night    p95(|P|) at POA < 5 W/m2 <= 1 % DC    (offset/metering faults)
  sensor   p99.9(POA) <= 1500 W/m2               (irradiance sensor faults)
  length   >= 4 years of days with >= 8 valid daylight hours (guideline: >= 4-5 y)
  drift    two references: hour-matched POA/ERA5-GHI and the POA/clear-sky upper envelope;
           flagged when both trends exceed 1 %/yr in the same direction (sensor drift would
           masquerade as degradation)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import external_sources as ex
from src.weather.open_meteo import reanalysis

ROOT = Path(__file__).resolve().parents[2]
TIER_A = ROOT / "data/raw/tier_a"
# Only descriptive labels are fixed here; coordinates, time zone and capacity come
# from the NREL PVDAQ catalogue, and the climate class from the Beck et al. raster.
# 1430-1433 follow the NREL catalogue names (1430 is the NREL Mesa single-axis tracker, not a roof array)
PVDAQ_LABELS = {"1430": "NREL Mesa 1-axis tracker, Golden CO", "1432": "NREL S and TF roof, Golden CO",
                "1433": "NREL RSF1 roof, Golden CO",
                "1423": "Henderson NV", "1199": "Cockeysville MD", "1200": "Linthicum Heights MD",
                "1201": "Cherry Hill NJ", "1202": "Cherry Hill NJ", "1203": "Wilmington DE", "1204": "Chester PA",
                "2107": "Farm Solar Array, Arbuckle CA", "9068": "SR_CO single-axis tracker, Kersey CO"}
TZ_OVERRIDES = {"2107": "America/Los_Angeles", "9068": "America/Denver"}   # catalogue gives codes, see prize module


def _catalogue_row(sid: str) -> dict:
    cat = pd.read_csv(ROOT / "data/reference/pvdaq/pvdaq_systems_20250729.csv").drop_duplicates("system_id")
    row = cat[cat.system_id == int(sid)].iloc[0]
    from src.climate.classify import classify_sites, verify_reference
    raster, codebook, _ = verify_reference(ROOT / "data/reference/climate/source.json")
    climate = classify_sites(pd.DataFrame([{"system_id": sid, "latitude": row.latitude, "longitude": row.longitude}]),
                             raster, codebook).climate_class.iloc[0]
    return {"lat": float(row.latitude), "lon": float(row.longitude), "dc_kw": float(row.dc_capacity_kW),
            "tz": TZ_OVERRIDES.get(sid, str(row.timezone_or_utc_offset)), "climate": climate,
            "tracking": row.tracking == "tracking"}      # metadata only; the analysis uses measured POA

def std_offset_h(tz: str) -> float:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return ZoneInfo(tz).utcoffset(datetime(2021, 1, 15)).total_seconds() / 3600 if "/" in tz else float(tz)


def era5_at(series: pd.Series, labels_utc: pd.DatetimeIndex) -> np.ndarray:
    """ERA5 (UTC hours, interval-start labels) at arbitrary UTC labels; labels between ERA5 hours
    (half-hour zones such as ACST, UTC+9:30) are linearly interpolated in time (gaps <= 2 h)."""
    src = series[~series.index.duplicated()]
    union = src.index.union(labels_utc.dropna())
    return src.reindex(union).interpolate(method="time", limit=2, limit_area="inside").reindex(labels_utc).to_numpy()


def era5_local(series: pd.Series, offset_h: float, index: pd.DatetimeIndex) -> pd.Series:
    """ERA5 onto the frame's local-standard-time stamps (fixed UTC offset, no DST)."""
    labels = (index - pd.Timedelta(hours=offset_h)).tz_localize("UTC")
    return pd.Series(era5_at(series, labels), index=index)


def detect_time_convention(hourly: pd.DataFrame, meta: dict) -> dict:
    """Which clock do the raw stamps follow? Each hypothesis (local standard, local wall clock with
    DST, UTC) maps the stamps to UTC; the one whose on-site POA correlates best with ERA5 GHI wins.
    local_standard vs local_dst differ only in the DST season, so that pair is decided on it."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = meta["tz"]
    zone = ZoneInfo(tz)
    dst = zone.utcoffset(datetime(2021, 1, 15)) != zone.utcoffset(datetime(2021, 7, 15))
    era5, _ = reanalysis(meta["lat"], meta["lon"], str(hourly.index.min().date()), str(hourly.index.max().date()),
                         ROOT / "data/cache/weather", variables=("shortwave_radiation",))
    poa = hourly.poa_wm2
    scores = {}
    for conv in (("local_standard", "local_dst", "utc") if dst else ("local_standard", "utc")):
        labels = to_utc(hourly.index, tz, conv)
        both = pd.DataFrame({"s": poa.to_numpy(), "e": era5_at(era5.shortwave_radiation, labels)}, index=labels)
        both = both[both.index.notna()].dropna()
        both = both[(both.s > 20) | (both.e > 20)]
        local = both.index.tz_convert(tz)
        in_dst = np.array([bool(t.dst()) for t in local]) if dst else np.zeros(len(both), bool)
        scores[conv] = {"r_all": float(both.s.corr(both.e)),
                        "r_dst_season": float(both[in_dst].s.corr(both[in_dst].e)) if in_dst.sum() > 500 else np.nan}
    rank = sorted(scores, key=lambda c: -scores[c]["r_all"])
    best, second = rank[0], rank[1]
    key = "r_dst_season" if {best, second} == {"local_standard", "local_dst"} else "r_all"
    if key == "r_dst_season" and scores[second][key] > scores[best][key]:
        best, second = second, best
    return {"time_convention": best, "convention_margin": round(scores[best][key] - scores[second][key], 4),
            "convention_r": round(scores[best]["r_all"], 4)}


def to_utc(raw: pd.DatetimeIndex, tz: str, convention: str) -> pd.DatetimeIndex:
    if convention == "utc":
        return raw.tz_localize("UTC")
    if convention == "local_standard":
        return (raw - pd.Timedelta(hours=std_offset_h(tz))).tz_localize("UTC")
    return raw.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def _hourly(native: pd.DataFrame, rename: dict) -> pd.DataFrame:
    frame = native.rename(columns=rename)
    keep = [c for c in ("power_w", "poa_wm2", "t_amb_c", "t_module_c") if c in frame]
    frame = frame[keep]
    hourly = frame.groupby(frame.index.floor("h")).mean()
    counts = frame.power_w.groupby(frame.index.floor("h")).size()
    step = max(1, int(round(frame.index.to_series().diff().dt.total_seconds().median() / 60)))
    hourly = hourly[counts >= int(np.ceil(0.9 * 60 / step))]
    return hourly


def load_system(system_id: str) -> tuple[pd.DataFrame, dict]:
    rename = {"onsite_poa_wm2": "poa_wm2", "onsite_ambient_temp_c": "t_amb_c", "onsite_module_temp_c": "t_module_c"}
    if system_id.startswith("dkasc_"):
        manifest = json.loads((ROOT / "data/raw/external/dkasc/manifest.json").read_text())
        name = next(k for k, v in manifest.items() if f"dkasc_{v['array']}" == system_id)
        native, site = ex.dkasc(name)
        meta = {"system_id": system_id, "label": site["name"], "climate": "BWh", "tz": site["tz_name"],
                "lat": site["latitude"], "lon": site["longitude"], "dc_kw": site["dc_capacity_kw"],
                "source": "DKASC", "tracking": False}
    else:
        files = sorted(TIER_A.glob(f"*{system_id}_*.parquet"), key=lambda p: p.stat().st_size)
        if not files:
            raise FileNotFoundError(system_id)
        native = pd.read_parquet(files[-1])
        info = _catalogue_row(system_id)
        meta = {"system_id": f"pvdaq_{system_id}", "label": PVDAQ_LABELS.get(system_id, system_id), **info,
                "source": "PVDAQ"}
    frame = _hourly(native, rename)
    # Every frame is re-stamped to local STANDARD time (fixed offset): solar-consistent, no DST gaps.
    meta["std_offset_h"] = std_offset_h(meta["tz"])
    if "poa_wm2" not in frame or frame.poa_wm2.notna().mean() < 0.3:
        meta["time_convention"] = "undetermined (no on-site POA)"       # the screen rejects such systems
        return frame, meta
    meta.update(detect_time_convention(frame, meta))
    utc = to_utc(frame.index, meta["tz"], meta["time_convention"])
    frame = frame[utc.notna()]
    frame.index = (utc[utc.notna()] + pd.Timedelta(hours=meta["std_offset_h"])).tz_localize(None)
    frame = frame[~frame.index.duplicated()].sort_index()
    meta["temperature_basis"] = "on-site"
    if "t_amb_c" not in frame and "t_module_c" not in frame:
        era5, _ = reanalysis(meta["lat"], meta["lon"], str(frame.index.min().date()), str(frame.index.max().date()),
                             ROOT / "data/cache/weather", variables=("temperature_2m",))
        frame = frame.assign(t_amb_c=era5_local(era5.temperature_2m, meta["std_offset_h"], frame.index))
        meta["temperature_basis"] = "ERA5 2-m temperature"
    return frame, meta


def screen(frame: pd.DataFrame, meta: dict) -> dict:
    cap = meta["dc_kw"] * 1000.0
    out = {"system_id": meta["system_id"], "climate": meta["climate"], "label": meta["label"],
           "first": str(frame.index.min().date()), "last": str(frame.index.max().date())}
    reasons = []
    if "poa_wm2" not in frame or frame.poa_wm2.notna().mean() < 0.3:
        return {**out, "pass": False, "reason": "no usable on-site POA"}
    p999 = float(frame.power_w.quantile(0.999))
    night = frame[frame.poa_wm2 < 5].power_w.abs()
    night95 = float(night.quantile(0.95)) if len(night) else 0.0
    poa999 = float(frame.poa_wm2.quantile(0.999))
    day = frame[(frame.poa_wm2 > 50) & frame.power_w.notna()]
    good_days = day.groupby(day.index.date).size()
    years = float((good_days >= 8).sum() / 365.25)
    out.update({"p999_over_dc": round(p999 / cap, 3), "night95_over_dc": round(night95 / cap, 4),
                "poa_p999": round(poa999, 1), "valid_years": round(years, 2)})
    if p999 > 1.10 * cap:
        reasons.append("power above 1.10 x DC rating")
    if night95 > 0.01 * cap:
        reasons.append("night power above 1 % of rating")
    if poa999 > 1500:
        reasons.append("POA sensor above 1500 W/m2")
    if years < 4:
        reasons.append("< 4 years of valid days")
    # Drift rule v2 (2026-09-28, set before any PLR of the affected systems was computed): two
    # independent references must agree -- ERA5 GHI and the pvlib clear-sky envelope -- each with a
    # trend beyond 1 %/yr in the same direction. ERA5 alone had flagged all three co-located Golden
    # systems, which points at the reference rather than three sensors drifting alike.
    try:
        drift = sensor_drift(frame, meta)
        out.update(drift)
    except Exception as exc:  # recorded, never dropped silently
        out["drift_check"] = f"ERA5 not computed: {type(exc).__name__}: {exc}"
    try:
        out.update(clearsky_drift(frame, meta))
    except Exception as exc:
        out["clearsky_check"] = f"not computed: {type(exc).__name__}: {exc}"
    era, cs = out.get("poa_era5_ratio_trend_per_year", np.nan), out.get("poa_clearsky_envelope_trend_per_year", np.nan)
    if np.isfinite(era) and np.isfinite(cs) and abs(era) > 0.01 and abs(cs) > 0.01 and np.sign(era) == np.sign(cs):
        reasons.append("POA sensor drift (> 1 %/yr vs ERA5 and vs clear-sky envelope)")
    return {**out, "pass": not reasons, "reason": "; ".join(reasons)}


MIN_DRIFT_MONTHS = 36        # the paper's Tier-A rule; the climate expansion lowers it to 24 for 2-3 y records (plan, 2026-10-01)


def sensor_drift(frame: pd.DataFrame, meta: dict, series: bool = False) -> dict:
    """Satellite/reanalysis cross-check of the on-site irradiance sensor (guideline: sensor drift).

    Hour-matched: only hours where both the sensor and ERA5 GHI exceed 50 W/m2 enter, so missing
    or thinned logging (e.g. DKASC after 2023) cannot bias the ratio. Monthly ratio of sums is
    divided by its calendar-month mean (removes the plane-of-array vs horizontal seasonal geometry);
    the trend of that anomaly (per year, relative) is the drift estimate."""
    era5, _ = reanalysis(meta["lat"], meta["lon"], str(frame.index.min().date()), str(frame.index.max().date()),
                         ROOT / "data/cache/weather", variables=("shortwave_radiation",))
    both = pd.DataFrame({"s": frame.poa_wm2,
                         "e": era5_local(era5.shortwave_radiation, meta["std_offset_h"], frame.index)}).dropna()
    both = both[(both.s > 50) & (both.e > 50)]
    month = both.index.to_period("M")
    sums = both.groupby(month).agg(s=("s", "sum"), e=("e", "sum"), n=("s", "size"))
    sums = sums[sums.n >= 100]
    ratio = sums.s / sums.e
    anomaly = ratio / ratio.groupby(ratio.index.month).transform("mean")
    t = np.array([(p.start_time - sums.index[0].start_time).days / 365.25 for p in sums.index])
    if len(anomaly) < MIN_DRIFT_MONTHS:
        raise ValueError(f"only {len(anomaly)} matched months")
    slope = np.polyfit(t, anomaly.to_numpy(), 1)[0]
    out = {"poa_era5_ratio_trend_per_year": round(float(slope), 4), "drift_months": int(len(anomaly))}
    return {**out, "_series": anomaly} if series else out


def clearsky_drift(frame: pd.DataFrame, meta: dict, series: bool = False) -> dict:
    """Second, model-based drift reference: the upper envelope (monthly 90th percentile) of
    POA_sensor / GHI_clear-sky (pvlib Ineichen, climatological Linke turbidity) on hours with
    GHI_cs > 300 W/m2. Clear hours dominate the upper tail; the calendar-month anomaly removes the
    plane-of-array geometry (and tracking), so no tilt/azimuth assumption enters the trend."""
    import pvlib
    labels = (frame.index - pd.Timedelta(hours=meta["std_offset_h"])).tz_localize("UTC")
    mid = labels + pd.Timedelta(minutes=30)
    cs = pvlib.location.Location(meta["lat"], meta["lon"]).get_clearsky(mid, model="ineichen").ghi.to_numpy()
    both = pd.DataFrame({"s": frame.poa_wm2.to_numpy(), "cs": cs}, index=frame.index).dropna()
    both = both[both.cs > 300]
    ratio = both.s / both.cs
    month = ratio.index.to_period("M")
    env = ratio.groupby(month).quantile(0.9)
    env = env[ratio.groupby(month).size() >= 100]
    if len(env) < MIN_DRIFT_MONTHS:
        raise ValueError(f"only {len(env)} months")
    anomaly = env / env.groupby(env.index.month).transform("mean")
    t = np.array([(p.start_time - env.index[0].start_time).days / 365.25 for p in env.index])
    out = {"poa_clearsky_envelope_trend_per_year": round(float(np.polyfit(t, anomaly.to_numpy(), 1)[0]), 4)}
    return {**out, "_series": anomaly} if series else out
