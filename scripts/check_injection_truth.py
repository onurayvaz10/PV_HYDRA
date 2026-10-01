"""H02 check: expected change of the first-year-normalized YoY rate under a multiplicative injection.

The injection (ml_plr.inject) multiplies power by (1 + r t / 100), t in years from the start of the cleaned record.
RdTools degradation_year_on_year recentres the daily series on its first-year median. For a record whose normalized
performance declines linearly, the exact expected change is

    dPLR_exact = r [1 + (d / 100)(2 t_c - 1)] / (1 + r / 200)

(d: the method's own rate on the unmodified record, t_c: median midpoint of the YoY pairs, years), whereas the
manuscript's Eq. (2) used the first-order form dPLR = r + 2 d r t_c / 100. This script runs the real pipeline
(inject -> RdTools sensor workflow) on noise-free PVWatts records built on real weather (real gaps) with known linear
rates and compares the observed change with both expressions.
Output: results/degradation_paper/tables/T26_injection_truth_check.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import ml_plr  # noqa: E402
from src.degradation.injection_truth import expected_change  # noqa: E402
from src.degradation.rdtools_pipeline import cell_temperature, sensor_plr  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

G = -0.0040
TAB = ROOT / "results/degradation_paper/tables"


def eq2(r, d, tc):
    return r + 2.0 * d * r * tc / 100.0


def exact(r, d, tc, t1=0.5, t1o=0.5):
    return expected_change(r, d, tc, t1, t1o)


def centring(f: pd.DataFrame) -> tuple[float, float]:
    """First-year median time of the reference daily series: from its first day and from the injection origin."""
    import rdtools
    from src.degradation.rdtools_pipeline import standard_mask
    poa = f.poa_wm2.clip(lower=0)
    tc = cell_temperature(poa, f.get("t_amb_c"), f.get("t_module_c"))
    exp = rdtools.normalization.pvwatts_dc_power(poa, 5000.0, temperature_cell=tc, gamma_pdc=G)
    power = f.power_w.clip(lower=0)
    norm = power / exp.replace(0, np.nan)
    m = standard_mask(norm, poa, tc, power)
    daily = rdtools.aggregation.aggregation_insol(norm[m], poa[m], frequency="D").dropna()
    daily = daily[daily > 0]
    start = daily.index.min()
    first = daily[daily.index <= start + pd.Timedelta("364D")]
    t1 = float(np.median((first.index - start).days.to_numpy() / 365.25))
    return t1, t1 + (start - f.index.min().floor("D")).days / 365.25


def record(frame: pd.DataFrame, d_true: float) -> pd.DataFrame:
    f = ml_plr.clean(frame)[["poa_wm2", "t_amb_c"] + (["t_module_c"] if "t_module_c" in frame else [])].copy()
    poa = f.poa_wm2.clip(lower=0)
    tc = cell_temperature(poa, f.get("t_amb_c"), f.get("t_module_c"))
    t = (f.index - f.index.min()).total_seconds() / (365.25 * 86400)
    f["power_w"] = (5000.0 * poa / 1000 * (1 + G * (tc - 25)) * (1 + d_true / 100 * t)).clip(lower=0).where(poa.notna())
    return f


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for raw in ("dkasc_7", "1423", "1430", "2107"):
        frame, meta = load_system(raw)
        for d_true in (0.0, -0.8, -2.0):
            base = record(frame, d_true)
            b = sensor_plr(base, 5.0, G)
            t1, t1o = centring(base)
            for r in (-0.5, -1.0, -2.0):
                inj = base.assign(power_w=ml_plr.inject(base, r))
                o = sensor_plr(inj, 5.0, G)
                obs = o["plr"] - b["plr"]
                rows.append({"weather": meta["system_id"], "d_true": d_true, "d_estimated": b["plr"], "r": r,
                             "t_center_years": b["t_center_years"], "observed_change": obs,
                             "eq2_first_order": eq2(r, b["plr"], b["t_center_years"]),
                             "exact_t05": exact(r, b["plr"], b["t_center_years"]),
                             "t1": t1, "t1_origin": t1o,
                             "exact": exact(r, b["plr"], b["t_center_years"], t1, t1o)})
                print(rows[-1]["weather"], d_true, r, round(obs, 4), "eq2", round(rows[-1]["eq2_first_order"], 4),
                      "exact", round(rows[-1]["exact"], 4), flush=True)
    out = pd.DataFrame(rows)
    out["err_eq2"] = out.observed_change - out.eq2_first_order
    out["err_exact"] = out.observed_change - out.exact
    out["err_exact_t05"] = out.observed_change - out.exact_t05
    out.to_csv(TAB / "T26_injection_truth_check.csv", index=False)
    print(out.groupby("r")[["err_eq2", "err_exact"]].agg(lambda e: np.abs(e).max()).round(4))
    print("max |obs - eq2| =", round(out.err_eq2.abs().max(), 4), " max |obs - exact(t1=0.5)| =",
          round(out.err_exact_t05.abs().max(), 4), " max |obs - exact(measured t1)| =", round(out.err_exact.abs().max(), 4))


if __name__ == "__main__":
    main()
