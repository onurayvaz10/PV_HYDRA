"""Common rate convention for scoring known-rate records (supervisor review, 3 October 2026).

Every estimator reports its rate relative to the first-year level (t = 0.5 yr): year-on-year rates are recentred on
the median of the first year, and kappa-HYDRA-D converts rho with r = rho / (1 + rho (0.5 - t_mid)). The generators
define their rate relative to the starting level (trend factor m(0) = 1). Scoring therefore converts the generator
truth to the first-year level:

    truth_first_year = truth_start / m(0.5)

with m the generator's trend factor. For a linear generator m(t) = 1 + r t / 100, so truth = r / (1 + r / 200);
for r = -1 %/yr this is -1.0050 %/yr, for r = -0.5 %/yr -0.5013 %/yr. On the ERA5 records (no gaps) the first-year
median of the year-on-year daily series lies at 0.5 yr.
"""

from __future__ import annotations

import numpy as np

# trend factor at t = 0.5 yr of the assumption-violating scenarios (run_semisynthetic_hard.multiplier)
SCENARIO_M05 = {"knee": 1 - 0.005 * 0.5, "step": 1 - 0.005 * 0.5, "soiling": 1 - 0.01 * 0.5, "non_pvwatts": 1 - 0.01 * 0.5}


def first_year(truth_start, m05=None):
    """Convert a rate relative to the starting level (%/yr) into the rate relative to the level at t = 0.5 yr."""
    truth_start = np.asarray(truth_start, dtype=float)
    m = 1 + truth_start / 200 if m05 is None else np.asarray(m05, dtype=float)
    return truth_start / m
