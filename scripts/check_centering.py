"""Actual first-year centring of the YoY rate on the measured records (reviewer point on Eq. 2).

RdTools recentres the insolation-weighted daily series on the median of its first 365 days. For a linearly declining
series this median sits at t_1, the median time of the first-year daily points (years from the injection origin, the
first cleaned hour). Eq. (2) in its general form is

    dPLR = r (1 + d (2 t_c - 2 t_1) / 100) / (1 + r t_1 / 100)

and reduces to the published form for t_1 = 0.5. This script measures t_1 and the offset of the daily series from the
injection origin on every retained system (reference filters) and reports the span T of the κ-HYDRA-D design data
(for the positivity bound |rho| < 2/T of the centred trend).
Output: results/degradation_paper/tables/T30_centering.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdtools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import kappa_hydra_d as kd  # noqa: E402
from src.degradation.ml_plr import clean  # noqa: E402
from src.degradation.rdtools_pipeline import cell_temperature, gamma_for, standard_mask  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

TAB = ROOT / "results/degradation_paper/tables"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ref = pd.read_csv(ROOT / "results/degradation_paper/reference_plr.csv")
    rows = []
    for _, s in ref[ref.status == "ok"].iterrows():
        raw = s.system_id.split("_", 1)[1] if s.system_id.startswith("pvdaq_") else s.system_id
        frame, meta = load_system(raw)
        f = clean(frame)
        origin = f.index.min()
        poa = f.poa_wm2.clip(lower=0)
        tcell = cell_temperature(poa, f.get("t_amb_c"), f.get("t_module_c"))
        expected = rdtools.normalization.pvwatts_dc_power(poa, meta["dc_kw"] * 1000.0, temperature_cell=tcell,
                                                          gamma_pdc=s.gamma)
        power = f.power_w.clip(lower=0)
        norm = power / expected.replace(0, np.nan)
        mask = standard_mask(norm, poa, tcell, power)
        daily = rdtools.aggregation.aggregation_insol(norm[mask], poa[mask], frequency="D").dropna()
        daily = daily[daily > 0]
        start = daily.index.min()
        first = daily[(daily.index >= start) & (daily.index <= start + pd.Timedelta("364D"))]
        t_rel = (first.index - start).days.to_numpy() / 365.25
        offset = (start - origin.floor("D")).days / 365.25
        design = kd.design(frame, meta["dc_kw"], s.gamma)
        rows.append({"system_id": s.system_id, "first_year_days": len(first),
                     "t1_from_daily_start": float(np.median(t_rel)), "daily_offset_years": offset,
                     "t1_from_origin": float(np.median(t_rel)) + offset, "t_center_years": s.get("t_center_years", np.nan),
                     "kappa_span_years": float(design.span.iloc[0]), "record_span_years": s.span_years})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T30_centering.csv", index=False)
    print(out.describe().round(3).to_string())


if __name__ == "__main__":
    main()
