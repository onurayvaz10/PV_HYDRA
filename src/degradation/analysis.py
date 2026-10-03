"""Estimate a site-level performance-loss trend from training data only.

This is a weather-normalised *system* performance trend, not a direct module
degradation measurement. Soiling, maintenance and hardware changes can confound it.
The protocol is based on year-on-year normalisation principles used by RdTools.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegradationEstimate:
    install_date: date
    train_start: date
    train_end: date
    annual_rate: float
    ci_low: float
    ci_high: float
    applied_rate: float
    monthly_pairs: int
    status: str


def installation_age_years(timestamps: pd.Series, install_date: date) -> pd.Series:
    """Calendar age is known at forecast issue time and does not use PV targets."""
    times = pd.to_datetime(timestamps, utc=True)
    age = (times - pd.Timestamp(install_date, tz="UTC")).dt.total_seconds() / (365.2425 * 86400)
    if (age < 0).any():
        raise ValueError("PV observation precedes documented installation date")
    return age.rename("installation_age_years")


def daily_weather_normalised_index(
    frame: pd.DataFrame, *, capacity_w: float, inverter_ac_limit_w: float,
    interval_minutes: int,
    temperature_coefficient_per_c: float, minimum_poa_wm2: float = 200,
    maximum_ac_fraction: float = 0.98, minimum_intervals: int = 24,
) -> pd.DataFrame:
    """Daily measured/expected energy on comparable unclipped daylight slots.

    Required fields: local_date, power_w, poa_irradiance_wm2, cell_temperature_c.
    The provided temperature coefficient must come from documented equipment
    metadata; it is never guessed from the entire forecast dataset.
    """
    required = {"local_date", "power_w", "poa_irradiance_wm2", "cell_temperature_c"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    if capacity_w <= 0 or inverter_ac_limit_w <= 0 or interval_minutes <= 0:
        raise ValueError("Capacity, inverter AC rating, and interval must be positive")
    if not -0.02 <= temperature_coefficient_per_c <= 0.0:
        raise ValueError("Documented temperature coefficient outside plausible range")
    data = frame.copy()
    for name in ("power_w", "poa_irradiance_wm2", "cell_temperature_c"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    keep = (data.poa_irradiance_wm2 >= minimum_poa_wm2) & (
        data.power_w >= 0) & (data.power_w < maximum_ac_fraction * inverter_ac_limit_w) & (
        data.cell_temperature_c.between(-30, 100))
    data = data.loc[keep].copy()
    thermal = 1 + temperature_coefficient_per_c * (data.cell_temperature_c - 25)
    data["expected_power_w"] = capacity_w * data.poa_irradiance_wm2 / 1000 * thermal
    data = data[data.expected_power_w > 0]
    daily = data.groupby("local_date", sort=True).agg(
        observed_energy_wh=("power_w", lambda values: float(values.sum() * interval_minutes / 60)),
        expected_energy_wh=("expected_power_w", lambda values: float(values.sum() * interval_minutes / 60)),
        valid_intervals=("power_w", "size"),
    )
    daily = daily[daily.valid_intervals >= minimum_intervals].copy()
    daily["performance_index"] = daily.observed_energy_wh / daily.expected_energy_wh
    daily = daily[np.isfinite(daily.performance_index) & (daily.performance_index > 0)]
    daily.index = pd.to_datetime(daily.index)
    return daily


def estimate_annual_loss(
    daily: pd.DataFrame, *, install_date: date, train_end: date,
    minimum_training_span_days: int = 1095,
    minimum_monthly_yoy_pairs: int = 18,
    minimum_valid_days_per_month: int = 10,
    bootstrap_resamples: int = 1000, bootstrap_seed: int = 20260927,
    plausible_annual_loss_range: tuple[float, float] = (-0.03, 0.0),
) -> DegradationEstimate:
    """Median matched-calendar-month YoY loss with bootstrap uncertainty.

    All input observations on or after train_end are discarded before fitting.
    A correction is applied only when the 95% CI is strictly negative and the
    rate lies in the registered plausible interval. Otherwise applied_rate=0.
    """
    if "performance_index" not in daily:
        raise ValueError("performance_index missing")
    if daily.empty:
        raise ValueError("No daily performance observations")
    data = daily.copy()
    data.index = pd.to_datetime(data.index)
    data = data.loc[(data.index < pd.Timestamp(train_end)) &
                    (data.index >= pd.Timestamp(install_date))]
    if data.empty:
        raise ValueError("No eligible training observations")
    train_start_date = data.index.min().date()
    train_end_date = data.index.max().date()
    if (train_end_date - train_start_date).days < minimum_training_span_days:
        return DegradationEstimate(install_date, train_start_date, train_end_date,
                                   float("nan"), float("nan"), float("nan"), 0.0, 0,
                                   "insufficient_training_span")
    data["year"] = data.index.year
    data["month"] = data.index.month
    monthly = data.groupby(["year", "month"]).agg(
        median_index=("performance_index", "median"),
        valid_days=("performance_index", "size"),
    )
    monthly = monthly[monthly.valid_days >= minimum_valid_days_per_month]
    changes = []
    for (year, month), row in monthly.iterrows():
        previous = (year - 1, month)
        if previous in monthly.index:
            earlier = monthly.loc[previous, "median_index"]
            if earlier > 0 and row.median_index > 0:
                changes.append(float(np.log(row.median_index / earlier)))
    if len(changes) < minimum_monthly_yoy_pairs:
        return DegradationEstimate(install_date, train_start_date, train_end_date,
                                   float("nan"), float("nan"), float("nan"), 0.0,
                                   len(changes), "insufficient_yoy_pairs")
    values = np.asarray(changes)
    annual_rate = float(np.expm1(np.median(values)))
    rng = np.random.default_rng(bootstrap_seed)
    block = min(12, len(values))
    blocks_needed = int(np.ceil(len(values) / block))
    starts = rng.integers(0, len(values) - block + 1,
                          size=(bootstrap_resamples, blocks_needed))
    sampled = np.concatenate([values[starts + offset] for offset in range(block)],
                             axis=1)[:, :len(values)]
    boot = np.expm1(np.median(sampled, axis=1))
    ci_low, ci_high = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    lower, upper = plausible_annual_loss_range
    accepted = lower <= annual_rate < upper and ci_high < 0
    status = "applied" if accepted else "uncertain_or_implausible"
    return DegradationEstimate(install_date, train_start_date, train_end_date,
                               annual_rate, ci_low, ci_high,
                               annual_rate if accepted else 0.0,
                               len(changes), status)


def drift_factor(timestamps: pd.Series, estimate: DegradationEstimate,
                 anchor_date: date | None = None) -> np.ndarray:
    """Known-calendar extrapolation from training-only fitted annual rate."""
    anchor = anchor_date or estimate.train_start
    years = (pd.to_datetime(timestamps, utc=True) - pd.Timestamp(anchor, tz="UTC")).dt.total_seconds().to_numpy() / (365.2425 * 86400)
    return np.power(1 + estimate.applied_rate, years)


def detrend_training_targets(power: np.ndarray, timestamps: pd.Series,
                             estimate: DegradationEstimate) -> np.ndarray:
    return np.asarray(power, dtype=float) / drift_factor(timestamps, estimate)


def restore_physical_predictions(prediction: np.ndarray, timestamps: pd.Series,
                                 estimate: DegradationEstimate) -> np.ndarray:
    return np.asarray(prediction, dtype=float) * drift_factor(timestamps, estimate)
