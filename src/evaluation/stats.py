"""Forecast metrics and significance tests on daylight targets.

* Diebold–Mariano with the Harvey–Leybourne–Newbold (1997) small-sample
  correction, doi:10.1016/S0169-2070(96)00719-4, applied to the per-issue
  mean absolute-error differential (multi-step horizon ``h`` = forecast length).
* Moving-block bootstrap CI of the relative nRMSE difference.
* Friedman rank test across sites.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def daylight_mask(samples) -> np.ndarray:
    return (samples.weight > 0) & (samples.daylight > 0)


def point_metrics(prediction: np.ndarray, samples, reference: np.ndarray | None = None,
                  mask: np.ndarray | None = None) -> dict[str, float]:
    mask = daylight_mask(samples) if mask is None else mask
    error = (prediction - samples.target)[mask]
    if error.size == 0:
        return {}
    mae, rmse = float(np.abs(error).mean()), float(np.sqrt((error ** 2).mean()))
    result = {"nMAE": mae, "nRMSE": rmse, "nMBE": float(error.mean()), "n": int(error.size)}
    actual = samples.target[mask]
    result["R2"] = float(1 - (error ** 2).sum() / max(((actual - actual.mean()) ** 2).sum(), 1e-12))
    if reference is not None:
        ref = (reference - samples.target)[mask]
        result["skill_RMSE"] = float(1 - rmse / np.sqrt((ref ** 2).mean()))
        result["skill_MAE"] = float(1 - mae / np.abs(ref).mean())
    return result


def per_issue_loss(prediction: np.ndarray, samples) -> np.ndarray:
    mask = daylight_mask(samples)
    absolute = np.abs(prediction - samples.target) * mask
    counts = mask.sum(axis=1)
    loss = np.where(counts > 0, absolute.sum(axis=1) / np.maximum(counts, 1), np.nan)
    return loss


def diebold_mariano_hln(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> dict[str, float]:
    """H0: equal expected loss. Negative statistic favours model a."""
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return {"dm_hln": float("nan"), "p_value": float("nan"), "n": n}
    mean = d.mean()
    lag = min(horizon - 1, n - 1)
    gamma = [np.mean((d[k:] - mean) * (d[:n - k] - mean)) for k in range(lag + 1)]
    variance = (gamma[0] + 2 * sum(gamma[1:])) / n
    if variance <= 0:
        return {"dm_hln": float("nan"), "p_value": float("nan"), "n": n}
    dm = mean / np.sqrt(variance)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = dm * correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return {"dm_hln": float(statistic), "p_value": float(p_value), "n": n,
            "mean_loss_difference": float(mean)}


def block_bootstrap_rmse_ratio(pred_a: np.ndarray, pred_b: np.ndarray, samples, block: int = 168,
                               resamples: int = 1000, seed: int = 20260927) -> dict[str, float]:
    """CI of nRMSE_a / nRMSE_b - 1 by resampling blocks of consecutive issues."""
    mask = daylight_mask(samples)
    sq_a = ((pred_a - samples.target) ** 2 * mask).sum(1)
    sq_b = ((pred_b - samples.target) ** 2 * mask).sum(1)
    counts = mask.sum(1)
    n = len(counts)
    rng = np.random.default_rng(seed)
    starts_max = max(n - block, 1)
    blocks = int(np.ceil(n / block))
    ratios = np.empty(resamples)
    for r in range(resamples):
        starts = rng.integers(0, starts_max, blocks)
        index = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n] % n
        ratios[r] = np.sqrt(sq_a[index].sum() / counts[index].sum()) / np.sqrt(
            sq_b[index].sum() / counts[index].sum()) - 1
    point = np.sqrt(sq_a.sum() / counts.sum()) / np.sqrt(sq_b.sum() / counts.sum()) - 1
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {"relative_nRMSE_difference": float(point), "ci95_low": float(low), "ci95_high": float(high)}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted((p, k) for k, p in p_values.items() if np.isfinite(p))
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (p, key) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def friedman(scores: dict[str, list[float]]) -> dict[str, float]:
    """scores[model] = metric per site (lower is better)."""
    models = list(scores)
    matrix = np.array([scores[m] for m in models]).T
    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "note": "needs >= 3 sites and >= 3 models"}
    statistic, p_value = stats.friedmanchisquare(*matrix.T)
    ranks = stats.rankdata(matrix, axis=1).mean(axis=0)
    return {"statistic": float(statistic), "p_value": float(p_value),
            "mean_rank": dict(zip(models, map(float, ranks)))}
