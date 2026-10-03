"""Second Arbuckle test of protocol P1: the fixed mask without the quantile clipping filter (inverter clipping is the
most likely second cause on this 893 kW plant). Reference YoY (typical and fitted coefficient) at every injected rate.
Output: results/fixed_mask/arbuckle_clipoff.csv (rates) and tables/R2c_arbuckle_clipoff.csv (recovery ratios)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_mask as fm  # noqa: E402
from analyze_revision import expected_change  # noqa: E402
from run_semisynthetic_gamma import fitted_gamma  # noqa: E402
from src.degradation.rdtools_pipeline import fixed_mask, sensor_plr  # noqa: E402

SID = "pvdaq_2107"


def main() -> None:
    base, meta, gamma, ym, _ = fm.load(SID)
    m_off = fixed_mask(base, meta["dc_kw"], gamma, clip=False)
    rows = []
    for inj in fm.INJECTIONS:
        frame = fm.injected(base, inj)
        for name, mask in (("clip on", ym), ("clip off", m_off)):
            r = sensor_plr(frame, meta["dc_kw"], gamma, mask=mask)
            g = fitted_gamma(frame, meta["dc_kw"])
            rf = sensor_plr(frame, meta["dc_kw"], g, mask=mask)
            rows.append({"mask": name, "injection": inj, "hours": int(mask.sum()), "plr_typical": r["plr"],
                         "t_c_typical": r["t_center_years"], "plr_fitted": rf["plr"], "t_c_fitted": rf["t_center_years"]})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(fm.OUT / "arbuckle_clipoff.csv", index=False)
    out = []
    for (name,), sub in d.groupby(["mask"]):
        z = sub[sub.injection == 0].iloc[0]
        for _, r in sub[sub.injection != 0].iterrows():
            for v in ("typical", "fitted"):
                e = expected_change(r.injection, z[f"plr_{v}"], z[f"t_c_{v}"])
                out.append({"mask": name, "coefficient": v, "injection": r.injection,
                            "recovery": (r[f"plr_{v}"] - z[f"plr_{v}"]) / e})
    t = pd.DataFrame(out).pivot_table(index=["mask", "coefficient"], columns="injection", values="recovery")
    t.to_csv(fm.OUT / "tables/R2c_arbuckle_clipoff.csv")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
