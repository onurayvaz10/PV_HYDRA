"""Tables of amendment 4 from the run files.

K2  inverter and fitted physical references on the 18 measured systems: injected-change error (taken from
    analyze_revision R1, which reads inverter_reference_runs.csv) and the change of the unmodified-record rate against
    the reference -> results/fixed_mask/tables/K2_measured_rate_change.csv
X   exploratory extension (US DOE RTC, Albuquerque, amendment 4b), same injection analysis as the main benchmark on
    results/fixed_mask_rtc -> results/fixed_mask_rtc/tables/X1_injection_by_method.csv, X2_fit_vs_rate.csv,
    X3_checks.json (pre-specified checks (i) and (ii) of amendment 4)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_revision as ar  # noqa: E402

EXT = ROOT / "results/fixed_mask_rtc"
SEVEN = ar.SEVEN


def k2_measured() -> None:
    f = ar.FM / "inverter_reference_runs.csv"
    rd = ar.read("rd_runs*.csv").drop_duplicates(["system_id", "injection"], keep="last")
    if not f.exists():
        return
    v = pd.read_csv(f).drop_duplicates(["system_id", "injection", "method"], keep="last")
    v = v[v.injection == 0].pivot(index="system_id", columns="method", values="plr")
    ref = rd[rd.injection == 0].set_index("system_id").plr
    out = v.sub(ref, axis=0).add_prefix("change_").join(v).join(ref.rename("reference"))
    out.to_csv(ar.TAB / "K2_measured_rate_change.csv")
    print(out.round(3).to_string())
    print(out.filter(like="change_").abs().agg(["median", "max"]).round(3).to_string())


def extension() -> None:
    original = ar.read
    ar.read = lambda pattern, base=None: original(pattern, EXT)
    ar.FM = EXT
    (EXT / "tables").mkdir(parents=True, exist_ok=True)
    e = ar.injection_table()
    e.to_csv(EXT / "tables/X0_injection_rows.csv", index=False)
    sys_mae = e.groupby(["method", "system_id"]).error.agg(lambda x: x.abs().mean()).unstack()
    t = pd.DataFrame({"mae": sys_mae.mean(axis=1), "n_systems": sys_mae.notna().sum(axis=1),
                      "n_rows": e.groupby("method").size()}).join(sys_mae).sort_values("mae")
    t.to_csv(EXT / "tables/X1_injection_by_method.csv")
    print(t.round(3).to_string())
    mlr = ar.read("ml_runs*.csv")
    checks = {"systems": sorted(e.system_id.unique()), "location": "Albuquerque, NM (BSk)", "label": "exploratory"}
    if len(mlr):
        fit = (mlr[mlr.injection == 0].drop_duplicates(["system_id", "model", "seed"], keep="last")
               .groupby("model").agg(val_rmse=("val_rmse", "mean"), val_r2=("val_r2", "mean"), n=("val_rmse", "size")))
        fit = fit.join(t.mae.rename("injection_mae"))
        fit.to_csv(EXT / "tables/X2_fit_vs_rate.csv")
        print(fit.round(4).to_string())
        seven = fit.loc[[m for m in SEVEN if m in fit.index]]
        best_fit = seven.val_rmse.idxmin()
        checks["lowest_rmse_model"] = best_fit
        checks["lowest_rmse_model_injection_rank"] = int(seven.injection_mae.rank().loc[best_fit])
        checks["check_i_lowest_rmse_is_most_accurate"] = bool(seven.injection_mae.idxmin() == best_fit)
        best_two = seven.injection_mae.idxmin()
        trio = {m: float(t.mae.get(m)) for m in ("reference", best_two, "khd") if m in t.index}
        checks["trio_mae"] = trio
        checks["check_ii_trio_within_delta"] = bool(len(trio) == 3 and max(trio.values()) - min(trio.values()) < ar.DELTA)
        checks["spread_seven_unmodified"] = (mlr[mlr.injection == 0].groupby(["system_id", "model"]).plr.mean().unstack()
                                             [[m for m in SEVEN if m in set(mlr.model)]]
                                             .pipe(lambda z: (z.max(axis=1) - z.min(axis=1))).round(3).to_dict())
    (EXT / "tables/X3_checks.json").write_text(json.dumps(checks, indent=1, default=float))
    print(json.dumps(checks, indent=1, default=float))


if __name__ == "__main__":
    k2_measured()
    extension()
