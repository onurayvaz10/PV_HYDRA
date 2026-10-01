"""Sensitivity of the reference to its temperature coefficient on the semi-synthetic records.

The main benchmark runs RdTools with the datasheet coefficient (G_DATASHEET), as in the standard sensor
workflow. A reviewer may ask whether a practitioner who fits the coefficient from the data, or who knew
the true one, would remove the bias of the thin-film-like response. Two variants on the same 48 records:

  rdtools_true_gamma    oracle: the generator's true coefficient (0.5 x datasheet for thin_film)
  rdtools_fitted_gamma  practical: gamma = b/a from the regression  P/(P_r POA/1000) = a + b (Tc - 25)
                        on the first 12 months, POA 600-1100 W/m2 (hours that pass the reference filters)

Output: results/degradation_paper/semisynthetic_gamma_runs.csv (one row per record and variant)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_semisynthetic import G_DATASHEET, RATES, SEEDS, WEATHER, synthesize  # noqa: E402
from src.degradation.rdtools_pipeline import cell_temperature, sensor_plr  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

OUT = ROOT / "results/degradation_paper/semisynthetic_gamma_runs.csv"
COLUMNS = ["weather", "climate", "family", "rate", "seed", "method", "gamma", "plr", "ci_low", "ci_high", "error"]


def fitted_gamma(syn: pd.DataFrame, dc_kw: float) -> float:
    poa = syn.poa_wm2
    tc = cell_temperature(poa.clip(lower=0), syn.get("t_amb_c"), syn.get("t_module_c"))
    first = syn.index < syn.index.min() + pd.DateOffset(months=12)
    sel = first & poa.between(600, 1100) & syn.power_w.notna() & tc.notna()
    y = syn.power_w[sel] / (1000 * dc_kw * poa[sel] / 1000)
    b, a = np.polyfit((tc[sel] - 25).to_numpy(), y.to_numpy(), 1)
    return float(b / a)


def main() -> None:
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    keys = set(zip(done.weather, done.family, done.rate.astype(float), done.seed, done.method))
    for raw in WEATHER:
        frame, meta = load_system(raw)
        wid = meta["system_id"]
        for family in ("pvwatts", "thin_film"):
            true_g = G_DATASHEET if family == "pvwatts" else 0.5 * G_DATASHEET
            for rate in RATES:
                for seed in SEEDS:
                    todo = [m for m in ("rdtools_true_gamma", "rdtools_fitted_gamma")
                            if (wid, family, rate, seed, m) not in keys]
                    if not todo:
                        continue
                    syn = synthesize(frame, meta, family, rate, seed)
                    rows = []
                    for method in todo:
                        g = true_g if method == "rdtools_true_gamma" else fitted_gamma(syn, 5.0)
                        r = sensor_plr(syn, 5.0, g)
                        rows.append({"weather": wid, "climate": meta["climate"], "family": family, "rate": rate,
                                     "seed": seed, "method": method, "gamma": g, "plr": r["plr"],
                                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "error": r["plr"] - rate})
                        print(wid, family, rate, seed, method, f"gamma={g:.5f}", round(r["plr"], 3), flush=True)
                    pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
