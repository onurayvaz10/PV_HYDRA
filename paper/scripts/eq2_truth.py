"""Expected change of a first-year-recentred year-on-year (YoY) rate under an injected linear trend (Eq. (2)).

A record whose normalized performance declines linearly with its own rate d (%/yr, relative to its first-year level
at time a) is multiplied by (1 + r_inj t / 100), t in years. A YoY pair centred at t_c changes by
d + r_inj + d r_inj (2 t_c - a) / 100, and the first-year level is scaled by (1 + r_inj a / 100), so the rate changes by

    delta_plr = r_inj [1 + d (2 t_c - 2 a) / 100] / (1 + r_inj a / 100)

which is Eq. (2) of the manuscript for a = 0.5 years. Supplementary Section E gives the derivation.
"""

from __future__ import annotations

import numpy as np


def expected_change(r_inj: float, d: float, t_c: float, a: float = 0.5) -> float:
    """Expected change (%/yr) of a YoY rate after multiplying power by (1 + r_inj t/100).

    r_inj : injected rate (%/yr); d : rate of the unmodified record (%/yr);
    t_c   : median midpoint of the YoY pairs (years from the first cleaned hour); a : first-year centre (years).
    """
    return r_inj * (1.0 + d * (2.0 * t_c - 2.0 * a) / 100.0) / (1.0 + r_inj * a / 100.0)


def first_order_change(r_inj: float, d: float, t_c: float) -> float:
    """First-order form r + 2 d r t_c / 100 (no recentring), shown in the paper to deviate by up to 0.058 %/yr."""
    return r_inj + 2.0 * d * r_inj * t_c / 100.0


def yoy_rate(t_years: np.ndarray, pi: np.ndarray) -> tuple[float, float]:
    """Noise-free YoY rate (%/yr) of a daily series, recentred on its first-year median, and the median pair midpoint.

    Pairs are values 365 days apart, as in the RdTools YoY estimator; the rate is the median of the annual changes
    divided by the first-year median level.
    """
    lag = 365
    first = np.median(pi[t_years < 1.0])
    change = (pi[lag:] - pi[:-lag]) / first * 100.0 * (365.0 / lag)
    mid = 0.5 * (t_years[lag:] + t_years[:-lag])
    order = np.argsort(change)
    k = len(change) // 2
    return float(change[order[k]]), float(mid[order[k]])
