"""Expected change of a rate under a multiplicative injection P * (1 + r t / 100) (t in years from the first cleaned
hour, ml_plr.inject). Shared by every summary script (Eq. (2) of the manuscript).

For a linearly declining normalized series and a year-on-year (YoY) estimator that recentres on the median of its
first year, the change of the rate is exactly

    dPLR = r (1 + d (2 t_c - 2 t_1) / 100) / (1 + r t_1' / 100)

  d     the method's own rate on the unmodified record (%/yr)
  t_c   median midpoint of the YoY pairs (years), t_1 the median time of the first-year daily points, both measured
        from the first day of the daily series (so their difference does not depend on the origin)
  t_1'  the same first-year median time measured from the injection origin (first cleaned hour)

For κ-HYDRA-D the first-year level is defined at t = 0.5 years from the injection origin by its rate conversion
r = rho / (1 + rho (0.5 - t_mid)), so t_1 = t_1' = 0.5 and t_c is the mean time of its data.
The per-system centring of the YoY methods comes from scripts/check_centering.py (tables/T30_centering.csv).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CENTERING = ROOT / "results/degradation_paper/tables/T30_centering.csv"


def expected_change(r: float, d: float, t_c: float, t1: float = 0.5, t1_origin: float = 0.5) -> float:
    return r * (1.0 + d / 100.0 * (2.0 * t_c - 2.0 * t1)) / (1.0 + r * t1_origin / 100.0)


@lru_cache(maxsize=1)
def _centering() -> pd.DataFrame | None:
    return pd.read_csv(CENTERING).set_index("system_id") if CENTERING.exists() else None


MEASURED = False   # main analysis: t_1 = 0.5 (T26: max deviation 0.006 %/yr vs 0.007 with the measured centring);
                   # True reproduces the sensitivity run with the measured centring of each system (T31)


def centering(system_id: str, method: str) -> tuple[float, float]:
    """(t_1 from the daily start, t_1 from the injection origin) for a method on a system."""
    if not MEASURED:
        return 0.5, 0.5
    c = _centering()
    if method.startswith("khd") or method.startswith("kappa") or c is None or system_id not in c.index:
        return 0.5, 0.5
    row = c.loc[system_id]
    return float(row.t1_from_daily_start), float(row.t1_from_origin)


def expected_for(method: str, system_id: str, r: float, d: float, t_c: float) -> float:
    t1, t1o = centering(system_id, method)
    return expected_change(r, d, t_c, t1, t1o)
