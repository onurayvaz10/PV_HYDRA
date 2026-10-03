"""Identifiability on the measured systems (protocol P6, committed 2026-10-02 before the runs).

For each of the 18 main systems (fixed hour mask of the unmodified record): free kappa-HYDRA-D fit (seed 0), then
rho fixed at 9 values within +-1 %/yr of the free fit, g refitted at each value, insolation-weighted pinball loss on
the training hours recorded. Profile width: range of the first-year rate over which a quadratic fitted to the
profile stays within 5 % of its minimum. Also the linear trends of the annual mean POA and ambient temperature.

Outputs: results/fixed_mask/profile_measured.csv (grid), profile_summary.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_mask as fm  # noqa: E402
from run_identifiability import pinball  # noqa: E402
from src.degradation import kappa_hydra_d as kd  # noqa: E402

GRID = fm.OUT / "profile_measured.csv"
SUMMARY = fm.OUT / "profile_summary.csv"


def main() -> None:
    hp = json.loads((ROOT / "configs/perf_models_hpo.json").read_text())["kappa_hydra_d"]["params"]
    done = set(pd.read_csv(GRID).system_id) if GRID.exists() else set()
    for sid in fm.systems():
        if sid in done:
            continue
        base, meta, gamma, ym, hh = fm.load(sid)
        data = kd.design(base, meta["dc_kw"], gamma, mask=ym & hh)
        t_mid = float(data.t_mid.iloc[0])
        to_first = lambda rho: rho / (1.0 + rho * (0.5 - t_mid))  # noqa: E731
        _, free_rho = kd._fit(data, 0, hp["hidden"], hp["lr"], hp["steps"], kd.DEVICE)
        rows = []
        for delta in np.linspace(-0.01, 0.01, 9):
            rho = free_rho + delta
            net, _ = kd._fit(data, 0, hp["hidden"], hp["lr"], hp["steps"], kd.DEVICE, fixed_rate=rho)
            rows.append({"system_id": sid, "rate_first_year_pct": 100 * to_first(rho),
                         "free_fit_rate_pct": 100 * to_first(free_rho), "train_pinball": pinball(net, data, kd.DEVICE)})
            print(sid, round(rows[-1]["rate_first_year_pct"], 3), rows[-1]["train_pinball"], flush=True)
        pd.DataFrame(rows).to_csv(GRID, mode="a", header=not GRID.exists(), index=False)
    summarize()


def weather_trends(base: pd.DataFrame) -> tuple[float, float]:
    """Linear trend of the mean POA (%/yr of its mean) and of the mean ambient temperature (K/yr). Records have
    incomplete years, so the trend is fitted to calendar-month anomalies (months with >= 20 days of data, anomaly
    against the mean of the same calendar month); this is the trend of the annual mean without the bias of
    partial years."""
    out = []
    for col in ("poa_wm2", "t_amb_c"):
        if col not in base or base[col].notna().sum() == 0:
            out.append(np.nan)
            continue
        s = base[col].clip(lower=0) if col == "poa_wm2" else base[col]
        days = s.groupby(s.index.to_period("M")).apply(lambda x: x.dropna().index.normalize().nunique())
        monthly = s.groupby(s.index.to_period("M")).mean()[days >= 20]
        anom = monthly - monthly.groupby(monthly.index.month).transform("mean")
        t = np.array([(p.year + (p.month - 0.5) / 12) for p in monthly.index])
        slope = np.polyfit(t, anom.to_numpy(), 1)[0]
        out.append(100 * slope / monthly.mean() if col == "poa_wm2" else slope)
    return float(out[0]), float(out[1])


def summarize() -> None:
    g = pd.read_csv(GRID)
    out = []
    for sid, p in g.groupby("system_id"):
        x, y = p.rate_first_year_pct.to_numpy(), p.train_pinball.to_numpy()
        a, b, c = np.polyfit(x, y, 2)
        xmin = -b / (2 * a) if a > 0 else x[np.argmin(y)]
        ymin = np.polyval([a, b, c], xmin)
        # quadratic within 5 % of its minimum: a (x - xmin)^2 = 0.05 ymin
        width = 2 * np.sqrt(0.05 * ymin / a) if a > 0 else np.nan
        single = bool(a > 0 and x.min() <= xmin <= x.max())
        base, meta, *_ = fm.load(sid)
        poa_tr, tamb_tr = weather_trends(base)
        out.append({"system_id": sid, "free_fit_rate_pct": float(p.free_fit_rate_pct.iloc[0]),
                    "profile_minimum_pct": float(xmin), "single_interior_minimum": single, "width_5pct": float(width),
                    "poa_trend_pct_per_year": float(poa_tr), "tamb_trend_k_per_year": float(tamb_tr)})
    pd.DataFrame(out).to_csv(SUMMARY, index=False)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summarize()
    else:
        main()
    sys.stdout.flush()
    os._exit(0)
