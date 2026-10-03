"""Two-stage ML PLR (RQ1): performance model on the reference period, PI, then YoY.

Stage 1  f(x) ~ P / P_rated, fitted on the first ``reference_months`` only, on
         daylight hours (POA >= 20 W/m2). Inside the reference period every 5th
         ISO week is held out for early stopping and model scoring (blocked, so
         all seasons are represented and adjacent hours do not leak).
Stage 2  PI = P / (P_rated * f(x)) over the whole record; the same filters,
         insolation-weighted daily aggregation and year-on-year estimator as the
         RdTools reference (rdtools_pipeline.yoy_from_normalized).

Synthetic check: P * (1 + r * t_years) injected over the whole record; the
recovered PLR change must equal r (in %/yr) for a correct method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.degradation import perf_models as pm
from src.degradation.rdtools_pipeline import cell_temperature, yoy_from_normalized

REFERENCE_MONTHS = 24


def analysis_start(frame: pd.DataFrame, min_share: float = 0.8) -> pd.Timestamp:
    """First day from which on-site POA covers >= 80 % of hours over the next 30 days.
    (e.g. DKASC POA sensors start in 2013 although power is logged from 2008)."""
    daily = frame.poa_wm2.notna().groupby(frame.index.floor("D")).mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0.0)
    ahead = daily[::-1].rolling(30, min_periods=30).mean()[::-1]
    ok = ahead[ahead >= min_share]
    return ok.index[0] if len(ok) else frame.index.min()


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated()]
    return frame[frame.index >= analysis_start(frame)]


def healthy_hours(frame: pd.DataFrame, dc_kw: float) -> pd.Series:
    """Training-target hygiene (not applied to the PI stage, which keeps the RdTools filters):
    daylight (POA >= 20 W/m2); hourly ratio P / (P_rated * POA/1000) in [0.01, 2] (the RdTools
    normalised-energy bounds: outages and meter faults); and whole outage/maintenance days removed,
    i.e. days whose ratio is < 30 % of the centred 30-day median (inverter trips, curtailment)."""
    poa = frame.poa_wm2
    ratio = frame.power_w.clip(lower=0) / (dc_kw * 1000.0 * poa.clip(lower=1) / 1000.0)
    day = poa >= 20
    hourly_ok = day & ratio.between(0.01, 2.0)
    daily = (frame.power_w.clip(lower=0).where(day).groupby(frame.index.floor("D")).sum()
             / (dc_kw * poa.where(day).groupby(frame.index.floor("D")).sum()).replace(0, np.nan))
    typical = daily.rolling(30, center=True, min_periods=10).median()
    outage_days = daily.index[(daily < 0.3 * typical).to_numpy()]
    return hourly_ok & ~frame.index.floor("D").isin(outage_days)


def reference_end(target: pd.Series, months: int = REFERENCE_MONTHS, min_hours: int = 150) -> pd.Timestamp:
    """End of the reference period: the first ``months`` data months, a data month being a calendar
    month with >= 150 healthy daylight hours (so gaps do not leave seasons unrepresented)."""
    counts = target.notna().groupby(target.index.to_period("M")).sum()
    good = counts[counts >= min_hours].index
    if len(good) < months:
        raise ValueError(f"only {len(good)} data months (< {months}) for the reference period")
    return good[months - 1].to_timestamp(how="end").floor("h") + pd.Timedelta(hours=1)


def prepare(frame: pd.DataFrame, dc_kw: float, reference_months: int = REFERENCE_MONTHS,
            healthy: pd.Series | None = None, full_record: bool = False) -> dict:
    """``healthy``: optional fixed target-hygiene mask (from the unmodified record, injection experiments);
    ``full_record``: train on the whole record instead of the reference period (information-equal variant;
    every fifth ISO week stays held out)."""
    frame = clean(frame)
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    # Inputs only: gaps of <= 2 h are interpolated so a single missing hour does not
    # void 24 windows; the target is never filled.
    features = pm.feature_frame(frame.reindex(grid), 0.0).interpolate(limit=2, limit_area="inside")
    ok = healthy_hours(frame, dc_kw) if healthy is None else healthy.reindex(frame.index, fill_value=False).astype(bool)
    target = (frame.power_w.clip(lower=0) / (dc_kw * 1000.0)).where(ok)
    x, y, stamps = pm.windows(features, target)
    ref_end = stamps.max() + pd.Timedelta(hours=1) if full_record else reference_end(pd.Series(y, index=stamps), reference_months)
    in_ref = (stamps < ref_end) & np.isfinite(y)
    val = in_ref & (stamps.isocalendar().week.to_numpy() % 5 == 0)
    train = in_ref & ~val
    return {"x": x, "y": y, "stamps": stamps, "train": train, "val": val, "ref_end": ref_end}


def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)), "nrmse_mean": rmse / float(y.mean())}


def plr_from_prediction(frame: pd.DataFrame, dc_kw: float, pred: np.ndarray, stamps: pd.DatetimeIndex,
                        power: pd.Series | None = None, soiling: bool = False, mask: pd.Series | None = None) -> dict:
    frame = clean(frame)
    power = (frame.power_w if power is None else power.reindex(frame.index)).clip(lower=0)
    expected = pd.Series(pred * dc_kw * 1000.0, index=stamps).reindex(frame.index)
    expected = expected.where(expected > 0.02 * dc_kw * 1000.0)
    poa = frame.poa_wm2.clip(lower=0)
    tcell = cell_temperature(poa, frame.get("t_amb_c"), frame.get("t_module_c"))
    return yoy_from_normalized(power / expected, poa, tcell, power, soiling, 60, mask=mask)


def inject(frame: pd.DataFrame, rate_pct_per_year: float) -> pd.Series:
    years = (frame.index - frame.index.min()).total_seconds() / (365.25 * 86400)
    return frame.power_w * (1 + rate_pct_per_year / 100.0 * years)
