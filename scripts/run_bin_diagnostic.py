"""Heterogeneity diagnostic (T10): YoY PLR of the RdTools PVWatts PI within POA and cell-temperature
bins, next to the pooled value, per system; plus the inter-annual spread of the hot-hour share.
A pooled PLR outside the range of its own bin PLRs indicates a composition (Simpson-type) effect.
Output: results/degradation_paper/tables/T10_bin_diagnostic.csv
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

POA_BINS = ((200, 450), (450, 700), (700, 950), (950, 1200))
T_BINS = ((0, 35), (35, 50), (50, 90))


def yoy(norm, poa, mask):
    daily = rdtools.aggregation.aggregation_insol(norm[mask], poa[mask], frequency="D").dropna()
    daily = daily[daily > 0]
    try:
        return float(rdtools.degradation.degradation_year_on_year(daily, confidence_level=95)[0])
    except Exception:
        return np.nan


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
        row = {"system_id": sid, "label": meta["label"], "climate": meta["climate"], "pooled": yoy(norm, poa, mask)}
        for lo, hi in POA_BINS:
            row[f"poa_{lo}_{hi}"] = yoy(norm, poa, mask & poa.between(lo, hi))
        for lo, hi in T_BINS:
            row[f"tcell_{lo}_{hi}"] = yoy(norm, poa, mask & tc.between(lo, hi))
        bins = [row[k] for k in row if k.startswith(("poa_", "tcell_")) and np.isfinite(row[k])]
        row["bin_min"], row["bin_max"] = (min(bins), max(bins)) if bins else (np.nan, np.nan)
        row["pooled_outside_bins"] = bool(bins) and not (row["bin_min"] <= row["pooled"] <= row["bin_max"])
        hot = (tc[mask & poa.between(600, 1000)] > 50)
        share = hot.groupby(hot.index.year).mean()
        row["hot_share_range"] = float(share.max() - share.min()) if len(share) else np.nan
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "label"}, flush=True)
    out = ROOT / "results/degradation_paper/tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "T10_bin_diagnostic.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
