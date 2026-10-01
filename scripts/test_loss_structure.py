"""Pre-specified test (DEGRADATION_PAPER_PLAN.md §8, 2026-09-29): across systems, the signed difference
kappa-HYDRA-D - RdTools correlates positively with the T13 tail index (losses concentrated in a subset of
hours make the insolation-weighted RdTools rate steeper than the median fit, i.e. the difference > 0).
Primary: one-sided Spearman rho > 0, alpha = 0.05. Secondary: the same with the non-linearity index.
Output: results/degradation_paper/tables/T14_loss_structure_test.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/degradation_paper"


def main() -> None:
    t13 = pd.read_csv(RES / "tables/T13_loss_structure.csv")
    ref = pd.read_csv(RES / "reference_plr.csv")[["system_id", "plr"]].rename(columns={"plr": "rdtools_plr"})
    kd = pd.read_csv(RES / "kappa_hydra_d_runs.csv")
    kd = kd[(kd.variant == "full") & (kd.injection == 0)][["system_id", "plr"]].rename(columns={"plr": "khd_plr"})
    d = t13.merge(ref, on="system_id").merge(kd, on="system_id")
    d["diff_khd_minus_rdtools"] = d.khd_plr - d.rdtools_plr
    rows = []
    for name, col, role in (("tail_index", "tail_index", "primary"), ("nonlinearity", "nonlinearity", "secondary")):
        rho, p_two = stats.spearmanr(d[col], d.diff_khd_minus_rdtools)
        p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
        rows.append({"test": name, "role": role, "n": len(d), "spearman_rho": rho, "p_one_sided": p_one,
                     "supported_at_0.05": bool(p_one < 0.05)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "tables/T14_loss_structure_test.csv", index=False)
    d.to_csv(RES / "tables/T14_loss_structure_data.csv", index=False)
    print(d[["system_id", "rdtools_plr", "khd_plr", "diff_khd_minus_rdtools", "tail_index", "nonlinearity"]].round(3).to_string(index=False))
    print(out.round(4).to_string(index=False))
    if len(d) < 15:
        print(f"NOTE: only {len(d)} of 15 systems available; the pre-specified test is defined on all 15.", file=sys.stderr)


if __name__ == "__main__":
    main()
