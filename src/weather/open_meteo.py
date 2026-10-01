"""Cached, keyless Open-Meteo access for reanalysis and archived lead-time NWP.

Sources (probed 2026-09-28):
* Historical Weather API, ``models=era5``: ECMWF ERA5 reanalysis, hourly,
  0.25 deg, gap-free; valid values reached 2026-09-22 at probe time.
* Previous Runs API: archived NWP stored by lead day (``*_previous_day1``,
  ``*_previous_day2``); complete from 2024-03-21, empty for 2023.

Open-Meteo radiation is the mean of the *preceding* hour. Project hourly
labels are interval starts (mean over [h, h+1)), so every returned series is
shifted back by one hour: the value stamped T describes [T-1h, T) and is
relabelled T-1h. Raw responses are cached with their SHA-256; identical
requests are never repeated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
REANALYSIS_VARIABLES = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
                        "temperature_2m", "cloud_cover", "relative_humidity_2m", "wind_speed_10m")
NWP_VARIABLES = ("shortwave_radiation", "cloud_cover", "temperature_2m")


def _request(url: str, params: dict, cache_dir: Path, retries: int = 6) -> tuple[dict, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({"url": url, **params}, sort_keys=True).encode()).hexdigest()[:32]
    path = cache_dir / f"{key}.json"
    if path.exists():
        raw = path.read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
        except requests.RequestException:          # dropped connection or proxy hiccup: wait and retry
            if attempt == retries - 1:
                raise
            time.sleep(15 * (attempt + 1))
            continue
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:                     # truncated body through the proxy: retry
                if attempt == retries - 1:
                    raise
                time.sleep(15 * (attempt + 1))
                continue
            if "hourly" not in payload:
                raise RuntimeError(f"Open-Meteo returned no hourly block: {payload}")
            raw = json.dumps(payload, sort_keys=True).encode()
            path.write_bytes(raw)
            (cache_dir / f"{key}.request.json").write_text(
                json.dumps({"url": url, "params": params}, indent=2), encoding="utf-8")
            return payload, hashlib.sha256(raw).hexdigest()
        if response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            time.sleep(65 if response.status_code == 429 else 10 * (attempt + 1))   # 429: per-minute quota
            continue
        raise RuntimeError(f"Open-Meteo HTTP {response.status_code}: {response.text[:300]}")
    raise RuntimeError("Open-Meteo request failed")


def _frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["hourly"])
    index = pd.to_datetime(frame.pop("time"), utc=True) - pd.Timedelta(hours=1)
    frame.index = pd.DatetimeIndex(index, name="label_utc")
    return frame.astype(float)


def reanalysis(lat: float, lon: float, start: str, end: str, cache_dir: Path,
               variables: tuple[str, ...] = REANALYSIS_VARIABLES) -> tuple[pd.DataFrame, dict]:
    """ERA5 hourly for [start, end] (UTC dates), one request per site-period."""
    params = {"latitude": round(float(lat), 4), "longitude": round(float(lon), 4),
              "start_date": start, "end_date": end, "hourly": ",".join(variables),
              "models": "era5", "timezone": "GMT"}
    payload, digest = _request(ARCHIVE_URL, params, cache_dir)
    frame = _frame(payload)
    meta = {"source": "Open-Meteo Historical Weather API, ERA5", "grid_latitude": payload.get("latitude"),
            "grid_longitude": payload.get("longitude"), "response_sha256": digest,
            "completeness": frame.notna().mean().round(4).to_dict()}
    return frame, meta


def previous_runs(lat: float, lon: float, start: str, end: str, cache_dir: Path,
                  variables: tuple[str, ...] = NWP_VARIABLES,
                  lead_days: tuple[int, ...] = (1, 2)) -> tuple[pd.DataFrame, dict]:
    """Archived NWP by lead day. Column ``<var>_previous_dayK`` for target hour
    tau comes from a run issued on UTC day date(tau) - K."""
    names = [f"{v}_previous_day{k}" for v in variables for k in lead_days]
    params = {"latitude": round(float(lat), 4), "longitude": round(float(lon), 4),
              "start_date": start, "end_date": end, "hourly": ",".join(names), "timezone": "GMT"}
    payload, digest = _request(PREVIOUS_RUNS_URL, params, cache_dir)
    frame = _frame(payload)
    meta = {"source": "Open-Meteo Previous Runs API", "response_sha256": digest,
            "completeness": frame.notna().mean().round(4).to_dict()}
    return frame, meta
