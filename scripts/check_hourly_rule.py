"""Effect of the hourly validity rule on the 18 main systems (supervisor review, 3 October 2026).

The rule keeps an hour when at least 90 % of its native intervals are present. The published runs counted rows
(groupby.size, rows with missing power included); the stated rule counts valid power samples (groupby.count).
For each system: hours kept under each rule, hours that fail the stated rule, and the reference rate (fixed mask
of each variant's own unmodified record). Output: results/fixed_mask/tables/R10_hourly_rule.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_mask as fm  # noqa: E402
from src.degradation import ml_plr  # noqa: E402
from src.degradation.rdtools_pipeline import fixed_mask, gamma_for, sensor_plr  # noqa: E402


def load(sid: str, rule: str):
    os.environ["PV_HOURLY_RULE"] = rule
    raw = sid.replace("pvdaq_", "")
    if sid in fm.EXPANSION:
        from src.degradation import expansion as ex
        frame, meta = ex.load(raw)
    else:
        from src.degradation.tier_a import load_system
        frame, meta = load_system(raw)
    base = ml_plr.clean(frame)
    base = base[base.power_w.notna()]
    gamma, _ = gamma_for(meta["label"])
    return base, meta, gamma


def main() -> None:
    rows = []
    for sid in fm.systems():
        b0, meta, gamma = load(sid, "rows")
        b1, _, _ = load(sid, "valid")
        r0 = sensor_plr(b0, meta["dc_kw"], gamma, mask=fixed_mask(b0, meta["dc_kw"], gamma))
        r1 = sensor_plr(b1, meta["dc_kw"], gamma, mask=fixed_mask(b1, meta["dc_kw"], gamma))
        rows.append({"system_id": sid, "hours_rows_rule": len(b0), "hours_valid_rule": len(b1),
                     "hours_failing_stated_rule": len(b0.index.difference(b1.index)),
                     "fraction_failing": len(b0.index.difference(b1.index)) / len(b0),
                     "plr_rows_rule": r0["plr"], "plr_valid_rule": r1["plr"], "plr_change": r1["plr"] - r0["plr"]})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    os.environ["PV_HOURLY_RULE"] = "rows"
    out = ROOT / "results/fixed_mask/tables/R10_hourly_rule.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    print("max fraction failing", d.fraction_failing.max(), "max |plr change|", d.plr_change.abs().max())


if __name__ == "__main__":
    main()
