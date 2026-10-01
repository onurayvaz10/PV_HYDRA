"""Established non-ML baselines for the degradation benchmark.

pvusa_plr        two-stage rate with the linear PVUSA performance model, P/P_r = G (a + b G + c T_amb)
                 (G = POA/1000; no wind channel), fitted on the same reference-period hours as the ML models;
                 the second stage is the identical filter + insolation-weighted daily + YoY step.
daily_normalized PVWatts-normalized, filtered, insolation-weighted daily series (the reference's own series).
stl_rate         trend step alternative: STL decomposition (period 365 d, robust) of the daily series; the rate
                 is the least-squares slope of the trend component relative to its first-year level.
changepoint_rate trend step alternative: continuous piecewise-linear fit with at most one breakpoint on monthly
                 means (breakpoint chosen by least squares, kept only if it lowers BIC); the rate is the average
                 annual change of the fitted line over the record, relative to its starting level.
All rates in %/yr, first-year convention (as RdTools YoY).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rdtools

from src.degradation import ml_plr
from src.degradation.rdtools_pipeline import cell_temperature, standard_mask


def pvusa_prediction(d: dict) -> np.ndarray:
    last = d["x"][:, -1, :]                       # features of the target hour: poa_n (POA/1000), tcell_n, tamb_n, ...
    g, tamb = last[:, 0], last[:, 2]
    design = np.column_stack([g, g * g, g * tamb])
    coef, *_ = np.linalg.lstsq(design[d["train"]], d["y"][d["train"]], rcond=None)
    return design @ coef


def pvusa_plr(frame: pd.DataFrame, dc_kw: float, d: dict | None = None) -> dict:
    d = d or ml_plr.prepare(frame, dc_kw)
    pred = pvusa_prediction(d)
    out = ml_plr.plr_from_prediction(frame, dc_kw, pred, d["stamps"])
    out["val_rmse"] = ml_plr.scores(pred[d["val"]], d["y"][d["val"]])["rmse"]
    return out


def daily_normalized(frame: pd.DataFrame, dc_kw: float, gamma: float) -> pd.Series:
    data = frame.sort_index()
    data = data[~data.index.duplicated()]
    poa = data.poa_wm2.clip(lower=0)
    tcell = cell_temperature(poa, data.get("t_amb_c"), data.get("t_module_c"))
    expected = rdtools.normalization.pvwatts_dc_power(poa, dc_kw * 1000.0, temperature_cell=tcell, gamma_pdc=gamma)
    power = data.power_w.clip(lower=0)
    normalized = power / expected.replace(0, np.nan)
    mask = standard_mask(normalized, poa, tcell, power)
    daily = rdtools.aggregation.aggregation_insol(normalized[mask], poa[mask], frequency="D").dropna()
    return daily[daily > 0]


def _years(index: pd.DatetimeIndex) -> np.ndarray:
    return (index - index.min()).days.to_numpy() / 365.25


def stl_rate(daily: pd.Series) -> float:
    from statsmodels.tsa.seasonal import STL
    cal = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz=daily.index.tz)
    series = daily.reindex(cal).interpolate(limit_direction="both")
    trend = STL(series.to_numpy(), period=365, robust=True).fit().trend
    t = _years(cal)
    slope, intercept = np.polyfit(t, trend, 1)
    return float(100 * slope / intercept)              # relative to the level at the start of the record


def changepoint_rate(daily: pd.Series) -> dict:
    monthly = daily.resample("MS").mean().dropna()
    monthly = monthly.groupby(monthly.index).first()
    t = _years(monthly.index)
    y = monthly.to_numpy()
    # remove the seasonal cycle with month-of-year means (as change-point PLR methods do before fitting)
    season = pd.Series(y, index=monthly.index).groupby(monthly.index.month).transform("mean").to_numpy()
    y = y - season + y.mean()
    n = len(y)

    def fit(design):
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        sse = float(np.sum((design @ coef - y) ** 2))
        return coef, sse

    base = np.column_stack([np.ones(n), t])
    coef0, sse0 = fit(base)
    best = (sse0, None, coef0)
    for tb in t[12:-12]:                                 # breakpoint at least one year from each end
        coef, sse = fit(np.column_stack([np.ones(n), t, np.clip(t - tb, 0, None)]))
        if sse < best[0]:
            best = (sse, tb, coef)
    bic = lambda sse, k: n * np.log(sse / n) + k * np.log(n)  # noqa: E731
    sse, tb, coef = best
    if tb is None or bic(sse, 4) >= bic(sse0, 2):
        tb, coef = None, coef0
    grid = np.array([0.0, t.max()])
    line = coef[0] + coef[1] * grid + (coef[2] * np.clip(grid - tb, 0, None) if tb is not None else 0)
    return {"plr": float(100 * (line[1] - line[0]) / line[0] / t.max()),
            "breakpoint_years": float(tb) if tb is not None else np.nan}
