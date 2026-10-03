"""Weather-normalised daily performance index from daily energy.

PI_d = E_d / (P_dc * H_poa,d), with H_poa,d the daily plane-of-array insolation
(kWh m-2) from hourly irradiance (ERA5 GHI via Erbs decomposition and
isotropic transposition, or measured tilted irradiance when available).
Temperature is not corrected here (no documented coefficients for the
residential systems); the year-on-year estimator compares the same calendar
month, which removes most seasonal temperature effects. Low-insolation days
and implausible indices are excluded before the estimator sees them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def poa_insolation_from_ghi(ghi_hourly: pd.Series, latitude: float, longitude: float,
                            tilt: float, azimuth: float, tz: str) -> pd.Series:
    """Daily POA insolation (kWh m-2) on local calendar days. ``ghi_hourly`` is
    indexed by UTC interval-start labels (mean over the hour)."""
    ghi = ghi_hourly.dropna().clip(lower=0)
    mid = ghi.index + pd.Timedelta(minutes=30)
    solpos = pvlib.solarposition.get_solarposition(mid, latitude, longitude)
    erbs = pvlib.irradiance.erbs(ghi.to_numpy(), solpos.zenith.to_numpy(), mid)
    dni_extra = pvlib.irradiance.get_extra_radiation(mid)
    poa = pvlib.irradiance.get_total_irradiance(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth,
                                                erbs["dni"], ghi.to_numpy(), erbs["dhi"],
                                                dni_extra=dni_extra, model="isotropic")["poa_global"]
    poa = pd.Series(np.clip(np.asarray(poa, dtype=float), 0, None), index=ghi.index).fillna(0)
    local_day = (ghi.index.tz_convert(tz)).date
    daily = poa.groupby(local_day).sum() / 1000.0
    counts = poa.groupby(local_day).size()
    daily = daily[counts >= 22]                       # near-complete days only
    daily.index = pd.to_datetime(daily.index)
    return daily


def performance_index(energy_kwh: pd.Series, insolation_kwh_m2: pd.Series, capacity_kw: float,
                      minimum_insolation: float = 2.0) -> pd.DataFrame:
    joined = pd.DataFrame({"energy_kwh": energy_kwh, "insolation": insolation_kwh_m2}).dropna()
    joined = joined[(joined.insolation >= minimum_insolation) & (joined.energy_kwh > 0)]
    joined["performance_index"] = joined.energy_kwh / (capacity_kw * joined.insolation)
    median = joined.performance_index.median()
    joined = joined[joined.performance_index.between(0.3 * median, 1.5 * median)]
    return joined
