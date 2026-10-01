"""Rate-interval coverage on the semi-synthetic records (exact true rate).

κ-HYDRA-D: 95 % interval from five model seeds + leave-one-year-out jackknife (run_semisynthetic_ci.py).
Reference and two-stage models: 95 % bootstrap interval of the year-on-year median (run_semisynthetic.py).
Coverage = share of records whose interval contains the true rate.
Also T17: the reference with the datasheet, the true and a fitted temperature coefficient
(run_semisynthetic_gamma.py).
Outputs: results/degradation_paper/tables/T16_semisynthetic_ci.csv, T17_semisynthetic_gamma.csv (+ printed summary)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/degradation_paper"
LABEL = {"khd_full": "κ-HYDRA-D (jackknife + seeds)", "rdtools": "RdTools YoY (bootstrap)",
         "tcn": "TCN two-stage (bootstrap)", "xgboost": "XGBoost two-stage (bootstrap)"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def gamma_table() -> None:
    """T17: the reference with the datasheet, the true (oracle) and a fitted temperature coefficient."""
    g = RES / "semisynthetic_gamma_runs.csv"
    if not g.exists():
        return
    ss = pd.read_csv(RES / "semisynthetic_runs.csv")
    base = ss[ss.method == "rdtools"].assign(method="rdtools_datasheet_gamma", gamma=np.nan)
    runs = pd.concat([base, pd.read_csv(g)], ignore_index=True)
    out = runs.groupby(["family", "method"], sort=False).agg(
        n=("error", "size"), bias=("error", "mean"), mae=("error", lambda e: np.abs(e).mean()),
        gamma_median=("gamma", "median"), gamma_min=("gamma", "min"), gamma_max=("gamma", "max")).reset_index()
    out.to_csv(RES / "tables/T17_semisynthetic_gamma.csv", index=False)
    print("\nReference with datasheet / true / fitted temperature coefficient:")
    print(out.round(4).to_string(index=False))
    fit = runs[runs.method == "rdtools_fitted_gamma"]
    if len(fit):
        print("\nfitted-gamma error by weather site (thin_film):")
        print(fit[fit.family == "thin_film"].groupby("weather").agg(bias=("error", "mean"), gamma=("gamma", "median")).round(4))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    gamma_table()
    parts = [pd.read_csv(p) for p in sorted(RES.glob("semisynthetic_ci_runs*.csv"))]
    if not parts:
        raise SystemExit("no semisynthetic_ci_runs*.csv yet")
    ci = pd.concat(parts, ignore_index=True).drop_duplicates(["weather", "family", "rate", "noise_seed"], keep="last")
    ci = ci.rename(columns={"noise_seed": "seed"})
    ci["method"] = "khd_full"
    ci["error"] = ci.plr - ci.rate
    ss = pd.read_csv(RES / "semisynthetic_runs.csv")
    ss = ss[ss.method != "khd_full"]
    both = pd.concat([ci[["method", "weather", "family", "rate", "seed", "plr", "ci_low", "ci_high", "error"]],
                      ss[["method", "weather", "family", "rate", "seed", "plr", "ci_low", "ci_high", "error"]]],
                     ignore_index=True)
    both["covered"] = (both.ci_low <= both.rate) & (both.rate <= both.ci_high)
    both["width"] = both.ci_high - both.ci_low
    rows = []
    for (m, fam), g in both.groupby(["method", "family"], sort=False):
        k, n = int(g.covered.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"method": m, "label": LABEL.get(m, m), "family": fam, "n": n, "covered": k,
                     "coverage": k / n, "coverage_lo": lo, "coverage_hi": hi,
                     "median_width": g.width.median(), "mae": g.error.abs().mean(), "bias": g.error.mean()})
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(LABEL)}
    out = out.sort_values(["family", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    (RES / "tables").mkdir(exist_ok=True)
    out.to_csv(RES / "tables/T16_semisynthetic_ci.csv", index=False)
    n_khd = int((both.method == "khd_full").sum())
    print(f"κ-HYDRA-D records with a jackknife interval: {n_khd} of 48")
    print(out.round(3).to_string(index=False))
    if n_khd:
        k = both[both.method == "khd_full"]
        print("\nκ-HYDRA-D coverage by weather site:")
        print(k.groupby(["family", "weather"]).covered.agg(["sum", "size"]).to_string())
        miss = k[~k.covered]
        if len(miss):
            print("\nmissed records:")
            print(miss[["weather", "family", "rate", "seed", "plr", "ci_low", "ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
