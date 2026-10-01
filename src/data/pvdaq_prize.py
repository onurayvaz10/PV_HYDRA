"""NREL PVDAQ 2023 Solar Data Prize systems (research-grade, measured weather).

Units were read from each system's metadata JSON (2026-09-28):
  2105 Maui     meter_ac_output (kW, avg), irradiance GHI (W/m2), ambient temp (C)
  2107 Arbuckle meter_revenue_grade_ac_output (kW, avg), POA (W/m2), ambient temp (F)
  9068 Kersey   sum of 8 inverter-module AC power channels (kW, avg), POA pad 1 (W/m2)
Only files whose time column is ``measured_on`` are used, so one site never
mixes time-stamp conventions (``utc_measured_on`` files are excluded). The
convention itself is detected later from clear-day alignment, as for every
other system. Values are never rescaled beyond the documented units.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.fs as fs

BASE = "oedi-data-lake/pvdaq/2023-solar-data-prize/{sid}_OEDI/data/"
PRIZE = {
    "2105": {"power": (["2105_meter_data.csv"], r"^meter_ac_output", 1000.0),
             "onsite_ghi_wm2": (["2105_irradiance_data.csv"], r"^irradiance_ghi", 1.0),
             "onsite_ambient_temp_c": (["2105_environment_1_data.csv"], r"^ambient_temp", 1.0)},
    "2107": {"power": (["2107_meter_15m_data.csv", "2107_meter_15m_data_2024.csv", "2107_meter_15m_data_2025.csv"],
                       r"^meter_revenue_grade_ac_output", 1000.0),
             "onsite_poa_wm2": (["2107_irradiance_data.csv", "2107_irradiance_data_2024.csv"], r"^poa_irradiance", 1.0),
             "onsite_ambient_temp_c": (["2107_environment_data.csv", "2107_environment_data_2024.csv"],
                                       r"^ambient_temperature", "F")},
    "9068": {"power": (["9068_ac_power_data.csv", "9068_ac_power_data_20240101_20250430.csv"],
                       r"^inverter_module_\d\.\d_ac_power", 1000.0),
             "onsite_poa_wm2": (["9068_irradiance_data.csv", "9068_irradiance_data_20240101_20250430.csv"],
                                r"^pyranometer_\(class_a\)_pad_1_poa_irradiance_\(w/m2\)", 1.0)},
    # 9069 Social Circle GA (climate expansion, channel rule fixed 2026-09-30 before any rate): revenue-grade
    # meter 1 total AC power (kW); meter 2 total kept only for the redundancy check in the expansion screen
    # (src/degradation/expansion.py); the first class-A pyranometer POA (02a), as pad 1 for 9068; weather station 01.
    "9069": {"power": (["9069_meter_data.csv"], r"^meter_1_ac_power_\(kw\)", 1000.0),
             "meter2_power_w": (["9069_meter_data.csv"], r"^meter_2_ac_power_\(kw\)", 1000.0),
             "onsite_poa_wm2": (["9069_irradiance_data.csv"], r"^pyranometer_\(class_a\)_02a_poa_irradiance", 1.0),
             "onsite_ambient_temp_c": (["9069_environment_data.csv"],
                                       r"^weather_station_01_ambient_temperature_\(sensor_1\)", 1.0)},
}


def _read(s3, sid: str, name: str, pattern: str) -> pd.Series:
    with s3.open_input_stream(BASE.format(sid=sid) + name) as handle:
        raw = handle.read()
    header = raw[:20000].decode("utf-8", "replace").splitlines()[0].split(",")
    if header[0] != "measured_on":
        raise ValueError(f"{name}: time column {header[0]!r} excluded (mixed conventions)")
    columns = [c for c in header[1:] if re.search(pattern, c)]
    if not columns:
        raise ValueError(f"{name}: no column matches {pattern}")
    frame = pd.read_csv(io.BytesIO(raw), usecols=["measured_on", *columns])
    frame["measured_on"] = pd.to_datetime(frame.measured_on, errors="coerce")
    frame = frame.dropna(subset=["measured_on"]).set_index("measured_on")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    series = values.sum(axis=1, min_count=len(columns)) if len(columns) > 1 else values.iloc[:, 0]
    return series.groupby(level=0).mean()


def acquire_prize(sid: str, months: int, cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"prize_{sid}_last{months}m.parquet"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        return pd.read_parquet(path), json.loads(meta_path.read_text(encoding="utf-8"))
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    columns, used = {}, {}
    for name, (files, pattern, scale) in PRIZE[sid].items():
        parts = []
        for file in files:
            parts.append(_read(s3, sid, file, pattern))
            used.setdefault(name, []).append(file)
        series = pd.concat(parts).groupby(level=0).mean().sort_index()
        series = (series - 32) * 5 / 9 if scale == "F" else series * scale
        columns[name if name != "power" else "power_w"] = series
    native = pd.DataFrame(columns["power_w"].rename("power_w")).dropna()
    extra = [n for n in columns if n != "power_w" and n.endswith("_power_w")]   # further meters (9069)
    if extra:     # their own timestamps are kept: a meter may log where the registered one does not
        stamps = native.index.union(pd.Index([t for n in extra for t in columns[n].dropna().index]).unique())
        native = native.reindex(stamps.sort_values())
    for name, series in columns.items():
        if name != "power_w":
            native[name] = series.reindex(native.index)
    end = native.index.max().normalize()
    native = native[native.index >= end - pd.DateOffset(months=months)]
    native.index.name = "timestamp_raw"
    deltas = native.index.to_series().diff().dt.total_seconds().div(60)
    interval = int(round(float(deltas[(deltas > 0) & (deltas <= 60)].median())))
    native.to_parquet(path)
    meta = {"system_id": sid, "files": used, "native_interval_minutes": interval, "rows": int(len(native)),
            "metric_rule": "prize_documented_units", "weather_channels": [c for c in native.columns if c != "power_w"],
            "source": "NREL PVDAQ 2023 Solar Data Prize (OEDI)",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return native, meta
