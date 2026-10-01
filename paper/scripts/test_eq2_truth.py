"""Noise-free test of Eq. (2): on a linear daily record with an injected trend, the observed change of the
first-year-recentred YoY rate equals expected_change(); the first-order form does not.

Run: python -m pytest test_eq2_truth.py   (or python test_eq2_truth.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eq2_truth import expected_change, first_order_change, yoy_rate  # noqa: E402


def record(d: float, years: float = 8.0, a: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(years * 365)) / 365.0
    return t, 1.0 + d / 100.0 * (t - a)


def test_noise_free_exact_change() -> None:
    worst_exact, worst_first = 0.0, 0.0
    for d in (-0.2, -0.8, -1.5, -2.5):
        t, pi = record(d)
        base, _ = yoy_rate(t, pi)
        for r in (-0.25, -0.5, -1.0, -2.0):
            obs, t_c = yoy_rate(t, pi * (1.0 + r * t / 100.0))
            a = float(np.median(t[t < 1.0]))                  # first-year median time of a monotone daily record
            exact = expected_change(r, base, t_c, a)
            worst_exact = max(worst_exact, abs((obs - base) - exact))
            worst_first = max(worst_first, abs((obs - base) - first_order_change(r, base, t_c)))
    assert worst_exact < 1e-3, worst_exact
    assert worst_first > 10 * worst_exact
    print(f"max |observed - Eq. (2)| = {worst_exact:.2e} %/yr; first-order form up to {worst_first:.3f} %/yr")


def test_reduces_to_eq2_at_half_year() -> None:
    r, d, tc = -1.0, -0.8, 4.2
    eq2 = r * (1 + d * (2 * tc - 1) / 100) / (1 + r / 200)
    assert abs(expected_change(r, d, tc, 0.5) - eq2) < 1e-12


if __name__ == "__main__":
    test_noise_free_exact_change()
    test_reduces_to_eq2_at_half_year()
    print("2 tests passed")
