"""Window-limited native PVDAQ acquisition (NREL OEDI, anonymous S3).

Only the chosen AC-power metric (and on-site weather channels when present)
is read, only for the requested months (hive partitions year/month/day).
The native series is cached once; 5-min / 15-min / hourly views are derived
locally from complete bins only. Missing samples are never filled.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq
import requests

METRICS_URL = ("https://oedi-data-lake.s3.amazonaws.com/pvdaq/parquet/metrics/"
               "metrics__system_{sid}__part000.parquet")
PREFIX = "oedi-data-lake/pvdaq/parquet/pvdata/system_id={sid}/"
WEATHER = {"poa_irradiance": "onsite_poa_wm2", "module_temp": "onsite_module_temp_c",
           "ambient_temp": "onsite_ambient_temp_c"}


def metrics(system_id: str) -> tuple[pd.DataFrame, str]:
    response = requests.get(METRICS_URL.format(sid=system_id), timeout=90)
    response.raise_for_status()
    return pq.read_table(io.BytesIO(response.content)).to_pandas(), hashlib.sha256(response.content).hexdigest()


def choose_power_metric(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Same precedence as the earlier PVDAQ hourly screen (exact total AC power,
    metered, W_avg, single channel, else summed components)."""
    names = table.sensor_name.fillna("").str.lower()
    common = table.common_name.fillna("").str.lower()
    candidates = table.loc[common.eq("ac power") | names.str.contains(r"ac_power|^w_avg$")].copy()
    if candidates.empty:
        raise ValueError("no measured AC power metric")
    lowered = candidates.sensor_name.fillna("").str.lower()
    for mask, rule in ((lowered.eq("ac_power"), "exact_ac_power_total"),
                       (lowered.str.contains("metered"), "metered_ac_power"),
                       (lowered.eq("w_avg"), "reported_watt_average")):
        if mask.any():
            return candidates.loc[mask].iloc[[0]], rule
    if len(candidates) == 1:
        return candidates, "single_ac_power_channel"
    component = candidates.loc[lowered.str.match(r"^(?:inv\d+_)?ac_power_?\d*(?:_kw|_kwac|_hw)?$")]
    component = component.loc[~component.sensor_name.str.lower().str.match(r"^ac_power_\d+_\d+$")]
    if component.empty:
        raise ValueError("ambiguous AC power channels")
    return component, "sum_component_ac_power_channels"


def available_day_range(system_id: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last day that actually have sample files in the parquet archive
    (catalogue timestamps can extend beyond the archived samples)."""
    import re
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    years = sorted(int(m.group(1)) for info in s3.get_file_info(fs.FileSelector(PREFIX.format(sid=system_id)))
                   if (m := re.search(r"year=(\d+)$", info.path)))
    if not years:
        return None
    days = []
    for year in (years[0], years[-1]):
        for info in s3.get_file_info(fs.FileSelector(PREFIX.format(sid=system_id) + f"year={year}/", recursive=True)):
            match = re.search(r"year=(\d+)/month=(\d+)/day=(\d+)/", info.path)
            if info.type == fs.FileType.File and match:
                days.append(pd.Timestamp(int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return (min(days), max(days)) if days else None


def _read_days(s3, system_id: str, start: pd.Timestamp, end: pd.Timestamp, metric_ids: set[int],
               workers: int = 24) -> pd.DataFrame:
    """One recursive listing per year, then concurrent reads of the day files
    (dataset discovery walks every directory sequentially and is ~50x slower)."""
    import re
    from concurrent.futures import ThreadPoolExecutor

    files = []
    for year in range(start.year, end.year + 1):
        try:
            infos = s3.get_file_info(fs.FileSelector(PREFIX.format(sid=system_id) + f"year={year}/", recursive=True))
        except FileNotFoundError:
            continue
        for info in infos:
            match = re.search(r"year=(\d+)/month=(\d+)/day=(\d+)/", info.path)
            if info.type == fs.FileType.File and match and info.path.endswith(".parquet"):
                day = pd.Timestamp(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                if start <= day <= end:
                    files.append(info.path)

    def read(path: str) -> pd.DataFrame:
        with s3.open_input_stream(path) as handle:
            table = pq.read_table(io.BytesIO(handle.read()))
        keep = [c for c in ("measured_on", "utc_measured_on", "metric_id", "value") if c in table.column_names]
        frame = table.select(keep).to_pandas()
        return frame[frame.metric_id.isin(metric_ids)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(read, files))
    if not parts:
        return pd.DataFrame(columns=["measured_on", "metric_id", "value"])
    return pd.concat(parts, ignore_index=True)


def acquire(system_id: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path,
            case_insensitive_weather: bool = False) -> tuple[pd.DataFrame, dict]:
    """Native local-time samples for [start, end]: power_w (+ on-site weather).
    ``case_insensitive_weather`` (climate expansion only) also matches weather channels such as
    'POA_irradiance'; the paper's Tier-A systems keep the exact-case match they were built with."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pvdaq_{system_id}_{start:%Y%m}_{end:%Y%m}.parquet"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        return pd.read_parquet(path), json.loads(meta_path.read_text(encoding="utf-8"))
    table, metric_sha = metrics(system_id)
    names = table.sensor_name.fillna("").str.lower()
    power_like = table.loc[table.common_name.fillna("").str.lower().eq("ac power") | names.str.contains(r"ac_power|^w_avg$")]
    if case_insensitive_weather:
        table = table.assign(sensor_name=table.sensor_name.where(~table.sensor_name.str.lower().isin(WEATHER),
                                                                 table.sensor_name.str.lower()))
    weather = table.loc[table.sensor_name.isin(WEATHER)]
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    raw = _read_days(s3, system_id, start, end,
                     set(power_like.metric_id.astype(int)) | set(weather.metric_id.astype(int)))
    # Summary channels (e.g. a computed ac_power) may exist in metadata but not in
    # the stored samples: the precedence rule is applied to channels actually present.
    present = set(raw.metric_id.unique())
    power, rule = choose_power_metric(power_like.loc[power_like.metric_id.isin(present)])
    weather = weather.loc[weather.metric_id.isin(present)]
    wanted = pd.concat([power, weather]).drop_duplicates("metric_id")
    time_basis = "measured_on"
    if "utc_measured_on" in raw and raw.utc_measured_on.notna().mean() > 0.99:
        raw["measured_on"] = raw.utc_measured_on
        time_basis = "utc_measured_on"
    raw = raw.drop(columns=[c for c in ("utc_measured_on",) if c in raw])
    raw["measured_on"] = pd.to_datetime(raw.measured_on, errors="coerce")
    raw = raw.dropna(subset=["measured_on", "value"])
    info = wanted.set_index("metric_id")
    raw["value"] = raw.value * raw.metric_id.map(info.calc_scale.fillna(1.0)) + raw.metric_id.map(info.calc_offset.fillna(0.0))
    power_ids = set(power.metric_id.astype(int))
    p = raw[raw.metric_id.isin(power_ids)]
    p = p.groupby(["measured_on", "metric_id"]).value.mean().unstack()
    native = pd.DataFrame({"power_w": p.sum(axis=1, min_count=len(power_ids))}).dropna()
    for metric_id, name in weather.set_index("metric_id").sensor_name.items():
        series = raw[raw.metric_id == metric_id].groupby("measured_on").value.mean()
        native[WEATHER[name]] = series.reindex(native.index)
    native = native[np.isfinite(native.power_w)].sort_index()
    native.index.name = "timestamp_raw"
    deltas = native.index.to_series().diff().dt.total_seconds().div(60)
    deltas = deltas[(deltas > 0) & (deltas <= 60)]
    interval = int(round(float(deltas.median()))) if len(deltas) else 60
    meta = {"system_id": system_id, "power_metrics": power.sensor_name.astype(str).tolist(),
            "power_units": power.units.astype(str).tolist() if "units" in power else [],
            "metric_rule": rule, "metric_metadata_sha256": metric_sha, "time_basis": time_basis,
            "weather_channels": sorted(WEATHER[n] for n in weather.sensor_name),
            "native_interval_minutes": interval, "rows": int(len(native)),
            "start": str(start.date()), "end": str(end.date()),
            "source": "NREL PVDAQ, OEDI data lake (s3://oedi-data-lake/pvdaq/parquet)"}
    native.to_parquet(path)
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return native, meta


def resample(native: pd.DataFrame, native_minutes: int, minutes: int) -> pd.DataFrame:
    """Interval-start labelled means. A bin is complete when it holds at least 90 %
    of the expected native samples (all 4 of 4 at 15-min, 11 of 12 at 5-min,
    54 of 60 at 1-min); incomplete bins are never used as targets."""
    step = max(native_minutes, 1)
    expected = max(1, minutes // step)
    grouped = native.groupby(native.index.floor(f"{minutes}min"))
    out = grouped.mean()
    out["source_sample_count"] = grouped.power_w.size()
    out["expected_samples"] = expected
    out["complete"] = out.source_sample_count >= int(np.ceil(0.9 * expected))
    out.index.name = "timestamp_raw"
    return out
