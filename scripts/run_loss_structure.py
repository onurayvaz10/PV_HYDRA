"""Loss-structure diagnostics per system (T13), on the RdTools PVWatts normalised series.

  yoy_plr        RdTools YoY (median of year-over-year changes)
  ols_plr        linear trend of the daily insolation-weighted PI (first-year level = 1)
  slope_first/second_half   linear trend on each half of the record  -> non-linearity = second - first
  tail_index     how much more the lower tail of hourly kappa fell than its median:
                 (q50_late/q50_early) - (q10_late/q10_early), early/late = first/last two full years.
                 > 0: losses concentrate in a subset of hours (shading, partial faults), which an
                 insolation-weighted mean (RdTools) counts and a median fit (kappa-HYDRA-D) largely ignores.
  hour_block_spread  late/early median-kappa ratio, max minus min over hour-of-day blocks (low-sun-angle losses)
Output: results/degradation_paper/tables/T13_loss_structure.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdtools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degradation import ml_plr
from src.degradation.rdtools_pipeline import cell_temperature, gamma_for, standard_mask
from src.degradation.tier_a import load_system


def slope(daily: pd.Series) -> float:
    if len(daily) < 200:
        return np.nan
    t = (daily.index - daily.index[0]).days.to_numpy() / 365.25
    return float(100 * np.polyfit(t, daily.to_numpy(), 1)[0])


def main() -> None:
    ref = pd.read_csv(ROOT / "results/degradation_paper/reference_plr.csv")
    rows = []
    for sid in ref[ref.status == "ok"].system_id:
        frame, meta = load_system(sid.replace("pvdaq_", ""))
        f = ml_plr.clean(frame)
        poa = f.poa_wm2.clip(lower=0)
        tc = cell_temperature(poa, f.get("t_amb_c"), f.get("t_module_c"))
        g, _ = gamma_for(meta["label"])
        exp = rdtools.normalization.pvwatts_dc_power(poa, meta["dc_kw"] * 1000, temperature_cell=tc, gamma_pdc=g)
        power = f.power_w.clip(lower=0)
        norm = power / exp.replace(0, np.nan)
        mask = standard_mask(norm, poa, tc, power)
        daily = rdtools.aggregation.aggregation_insol(norm[mask], poa[mask], frequency="D").dropna()
        daily = daily[daily > 0]
        daily = daily / daily.iloc[:365].median()
        rd, _, _ = rdtools.degradation.degradation_year_on_year(daily, confidence_level=95)
        mid = daily.index[0] + (daily.index[-1] - daily.index[0]) / 2
        k = norm[mask]
        years = pd.Series(k.index.year, index=k.index)
        counts = years.value_counts().sort_index()
        full = counts[counts >= 0.5 * counts.max()].index      # years with at least half the typical data
        early, late = full[:2], full[-2:]
        ke, kl = k[years.isin(early)], k[years.isin(late)]
        tail = (kl.median() / ke.median()) - (kl.quantile(0.1) / ke.quantile(0.1))
        blocks = pd.cut(k.index.hour, [0, 9, 12, 15, 24], labels=["<9", "9-12", "12-15", ">15"], right=False)
        be = k[years.isin(early)].groupby(blocks[years.isin(early).to_numpy()], observed=True).median()
        bl = k[years.isin(late)].groupby(blocks[years.isin(late).to_numpy()], observed=True).median()
        ratio = (bl / be).dropna()
        rows.append({"system_id": sid, "label": meta["label"], "climate": meta["climate"], "yoy_plr": float(rd),
                     "ols_plr": slope(daily), "slope_first_half": slope(daily[daily.index < mid]),
                     "slope_second_half": slope(daily[daily.index >= mid]),
                     "tail_index": float(tail), "hour_block_spread": float(ratio.max() - ratio.min()) if len(ratio) else np.nan,
                     "early_years": f"{early[0]}-{early[-1]}", "late_years": f"{late[0]}-{late[-1]}"})
        rows[-1]["nonlinearity"] = rows[-1]["slope_second_half"] - rows[-1]["slope_first_half"]
        print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in rows[-1].items() if k2 != "label"}, flush=True)
    out = ROOT / "results/degradation_paper/tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "T13_loss_structure.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
