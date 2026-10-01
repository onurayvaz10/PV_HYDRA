"""Why the soiling scenario biases the rates: the same soiling realization (seed 0) on records with no trend and
with -1 %/yr, and a -1 %/yr record without soiling, for each of the four weather sites. RdTools YoY (median of
year-on-year changes) is compared with STL (least-squares trend of the decomposed series).
Output: results/degradation_paper/soiling_diagnostic.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_semisynthetic_hard as h  # noqa: E402
from src.degradation import baselines  # noqa: E402
from src.degradation.rdtools_pipeline import sensor_plr  # noqa: E402
from src.degradation.tier_a import load_system  # noqa: E402

OUT = ROOT / "results/degradation_paper/soiling_diagnostic.csv"


def main() -> None:
    orig = h.multiplier
    rows = []
    for raw in h.WEATHER:
        frame, meta = load_system(raw)
        for case, rate, soil_on in [("soiling, no trend", 0.0, True), ("-1 %/yr, no soiling", -1.0, False),
                                    ("-1 %/yr with soiling", -1.0, True)]:
            def m(scenario, index, seed, rate=rate, soil_on=soil_on):
                _, s = orig("soiling", index, seed)
                t = (index - index.min()).total_seconds().to_numpy() / (365.25 * 86400)
                return 1 + rate / 100 * t, (s if soil_on else np.ones_like(s))
            h.multiplier = m
            syn, _ = h.synthesize(frame, meta, "soiling", 0)
            daily = baselines.daily_normalized(syn, 5.0, h.G_DATASHEET)
            rows.append({"weather": meta["system_id"], "case": case, "true_rate": rate,
                         "rdtools_yoy": sensor_plr(syn, 5.0, h.G_DATASHEET)["plr"], "stl": baselines.stl_rate(daily)})
            print(rows[-1], flush=True)
        h.multiplier = orig
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
