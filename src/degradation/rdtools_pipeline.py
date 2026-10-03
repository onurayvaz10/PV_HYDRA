"""Reference PLR with NREL RdTools (sensor workflow) — the paper's reference method.

Steps (each filter is reported with its reason in the paper):
  1. Cell temperature: measured module temperature + POA/1000 * 3 K (SAPM module-to-cell
     offset), else SAPM model from ambient temperature and wind (open-rack glass/polymer).
  2. Expected DC-equivalent power: RdTools PVWatts, P_exp = P_rated * POA/1000 * (1 + gamma*(T_cell-25)).
  3. Filters: POA 200-1200 W/m2 (low-irradiance and sensor-saturation noise), cell temperature
     -50..110 C (sensor faults), clipping (RdTools quantile clip filter: inverter limit is not
     degradation), normalised energy 0.01..2 (outages and meter faults).
  4. Insolation-weighted daily aggregation, then year-on-year degradation with 95 % CI.
  5. Optional soiling (stochastic rate and recovery) so that soiling losses are separated.
gamma (power temperature coefficient) comes from the technology when no datasheet exists:
typical datasheet values, recorded as an assumption and varied in a sensitivity analysis.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pvlib
import rdtools
import rdtools.soiling  # experimental module; must be imported explicitly

GAMMA_BY_TECHNOLOGY = {"mono-si": -0.0045, "poly-si": -0.0045, "cdte": -0.0032, "amorphous": -0.0020,
                       "a-si": -0.0020, "hit": -0.0029, "cigs": -0.0036, "default": -0.0040}


def gamma_for(label: str) -> tuple[float, str]:
    text = (label or "").lower()
    for key, value in GAMMA_BY_TECHNOLOGY.items():
        if key != "default" and key in text:
            return value, f"typical {key} datasheet value (assumption)"
    return GAMMA_BY_TECHNOLOGY["default"], "generic c-Si value (assumption)"


def cell_temperature(poa: pd.Series, t_amb: pd.Series | None, t_module: pd.Series | None,
                     wind: pd.Series | None = None) -> pd.Series:
    if t_module is not None and t_module.notna().mean() > 0.5:
        return t_module + poa / 1000.0 * 3.0
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    wind = wind if wind is not None else pd.Series(1.0, index=poa.index)
    return pvlib.temperature.sapm_cell(poa, t_amb, wind.fillna(1.0), **params)


def sensor_plr(frame: pd.DataFrame, dc_kw: float, gamma: float, soiling: bool = False,
               freq_minutes: int = 60, mask: pd.Series | None = None) -> dict:
    """``frame``: regular time index; columns power_w, poa_wm2 and t_amb_c and/or t_module_c.
    ``mask``: optional fixed filter mask (see fixed_mask); None keeps the standard per-record mask."""
    data = frame.copy().sort_index()
    data = data[~data.index.duplicated()]
    poa = data.poa_wm2.clip(lower=0)
    tcell = cell_temperature(poa, data.get("t_amb_c"), data.get("t_module_c"), data.get("wind_ms"))
    expected = rdtools.normalization.pvwatts_dc_power(poa, dc_kw * 1000.0, temperature_cell=tcell,
                                                      gamma_pdc=gamma)
    power = data.power_w.clip(lower=0)
    normalized = power / expected.replace(0, np.nan)
    return yoy_from_normalized(normalized, poa, tcell, power, soiling, freq_minutes, mask=mask)


def fixed_mask(frame: pd.DataFrame, dc_kw: float, gamma: float, clip: bool = True) -> pd.Series:
    """Filter mask computed once on the unmodified record (reference normalization), to be applied to every
    injected rate and every method, so that all estimators are scored on identical hours. Without it the
    power-dependent clipping filter removes different hours at each injected rate (fixed 2026-10-02)."""
    data = frame.copy().sort_index()
    data = data[~data.index.duplicated()]
    poa = data.poa_wm2.clip(lower=0)
    tcell = cell_temperature(poa, data.get("t_amb_c"), data.get("t_module_c"), data.get("wind_ms"))
    expected = rdtools.normalization.pvwatts_dc_power(poa, dc_kw * 1000.0, temperature_cell=tcell,
                                                      gamma_pdc=gamma)
    power = data.power_w.clip(lower=0)
    return standard_mask(power / expected.replace(0, np.nan), poa, tcell, power, clip).fillna(False).astype(bool)


def standard_mask(normalized: pd.Series, poa: pd.Series, tcell: pd.Series, power: pd.Series,
                  clip: bool = True) -> pd.Series:
    mask = (rdtools.filtering.poa_filter(poa, 200, 1200)
            & rdtools.filtering.tcell_filter(tcell, -50, 110)
            & rdtools.filtering.normalized_filter(normalized, 0.01, 2.0))
    try:
        if clip:                                     # clip=False: second Arbuckle test of protocol P1
            mask &= rdtools.filtering.clip_filter(power, model="quantile")
    except Exception:  # short or flat series: clipping filter not applicable, recorded
        pass
    return mask


def yoy_from_normalized(normalized: pd.Series, poa: pd.Series, tcell: pd.Series, power: pd.Series,
                        soiling: bool = False, freq_minutes: int = 60, mask: pd.Series | None = None) -> dict:
    """Shared second stage: identical filters, aggregation and YoY for every performance model.
    With ``mask`` (fixed_mask of the unmodified record) the hours are not recomputed from this series."""
    if mask is None:
        mask = standard_mask(normalized, poa, tcell, power)
    else:
        mask = mask.reindex(normalized.index, fill_value=False).astype(bool) & normalized.notna() & (normalized > 0)
    kept = float(mask.mean())
    insolation = poa * freq_minutes / 60.0
    daily = rdtools.aggregation.aggregation_insol(normalized[mask], insolation[mask], frequency="D")
    daily = daily.dropna()
    daily = daily[daily > 0]
    result = {"points_kept_fraction": kept, "days": int(len(daily)),
              "span_years": float((daily.index.max() - daily.index.min()).days / 365.25) if len(daily) else 0.0}
    if result["span_years"] < 2.0 or len(daily) < 365:
        return {**result, "plr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "status": "insufficient span"}
    rd, ci, info = rdtools.degradation.degradation_year_on_year(daily, confidence_level=95)
    if os.environ.get("PV_RETURN_YOY") == "1":          # year-on-year slopes for interval sensitivity analyses
        result["_yoy_values"] = info.get("YoY_values")
    # Time centre of the YoY pairs (years from the first day): needed for the second-order term of
    # multiplicative injections, E(t) - E(t-1) = d + r + 2 d r t_c (first-year-normalised YoY).
    t = (daily.index - daily.index.min()).days.to_numpy() / 365.25
    result["t_center_years"] = float(np.median(t[t >= 1]) - 0.5) if (t >= 1).any() else np.nan
    result.update({"plr": float(rd), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "status": "ok",
                   "yoy_values": int(len(info["YoY_values"])) if "YoY_values" in info else None})
    if soiling:
        try:
            # SRR needs a regular daily calendar (missing days as NaN, tolerated up to day_scale).
            calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz=daily.index.tz)
            daily_regular = daily.reindex(calendar)
            daily_insol = insolation[mask].resample("D").sum().reindex(calendar)
            srr = rdtools.soiling.soiling_srr(daily_regular, daily_insol, reps=300)
            result.update({"soiling_ratio_median": float(srr[0]),
                           "soiling_ci": [float(srr[1][0]), float(srr[1][1])]})
        except Exception as exc:
            result.update({"soiling_status": f"not estimated: {type(exc).__name__}"})
    return result
