"""Sensitivity of the reference 95 % interval to serial dependence (amendment 3).

RdTools' default interval resamples the year-on-year slopes independently. Neighbouring slopes share weather and
soiling, so the interval may be too narrow. Here the same slopes, in time order, are resampled with a circular block
bootstrap (30-day blocks, 2000 resamples) and the 95 % interval of their median is compared with RdTools' default
interval, on the 18 main systems (unmodified record, fixed mask). RdTools' own 'circular_block' option needs a
gap-free daily calendar and is not used on these gapped records. Output: results/fixed_mask/tables/R11_block_bootstrap.csv
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
import run_fixed_mask as fm  # noqa: E402
from src.degradation.rdtools_pipeline import sensor_plr  # noqa: E402

BLOCK, REPS = 30, 2000


def block_interval(slopes: pd.Series, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the median slope; circular blocks of BLOCK consecutive values (values are near-daily)."""
    v = slopes.sort_index().to_numpy()
    n = len(v)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / BLOCK))
    meds = np.empty(REPS)
    for i in range(REPS):
        starts = rng.integers(0, n, k)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
        meds[i] = np.median(v[idx])
    return tuple(float(x) for x in np.quantile(meds, [0.025, 0.975]))


def main() -> None:
    os.environ["PV_RETURN_YOY"] = "1"
    rows = []
    for sid in fm.systems():
        base, meta, gamma, ym, _ = fm.load(sid)
        r = sensor_plr(base, meta["dc_kw"], gamma, mask=ym)
        lo, hi = block_interval(r["_yoy_values"])
        out = {"system_id": sid, "plr": r["plr"], "ci_low_default": r["ci_low"], "ci_high_default": r["ci_high"],
               "ci_low_block": lo, "ci_high_block": hi, "width_default": r["ci_high"] - r["ci_low"], "width_block": hi - lo}
        out["width_ratio"] = out["width_block"] / out["width_default"]
        rows.append(out)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/fixed_mask/tables/R11_block_bootstrap.csv", index=False)
    print("median width default", d.width_default.median(), "block", d.width_block.median(), "median ratio", d.width_ratio.median())


if __name__ == "__main__":
    main()
