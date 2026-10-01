"""Forward (later-period) and held-out-location validation, recomputed from the raw run files of
run_independent_validation.py and run_independent_transfer.py (results/independent_validation_20260929).

Weighting: locations count equally (the 12 DKASC arrays share one site). Each model/array is first averaged over its
five seeds, then over the arrays of a location, then over the four locations.
Outputs (results/degradation_paper/tables/):
  T22_forward_by_model.csv        later-period RMSE, MAE, R2 and kappa-HYDRA-D 80 % hourly coverage
  T23_heldout_location.csv        held-out-location RMSE by model and condition
  T24_field_rate_sensitivity.csv  field-rate sensitivity to the temperature coefficient (+/-20 %), if available
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IV = ROOT / "results/independent_validation_20260929"
TAB = ROOT / "results/degradation_paper/tables"


def equal_location(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    per_array = df.groupby(keys + ["location", "system_id"])[cols].mean().reset_index()
    per_loc = per_array.groupby(keys + ["location"])[cols].mean().reset_index()
    return per_loc.groupby(keys)[cols].mean().reset_index()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    c = pd.read_csv(IV / "chronological_runs.csv")
    assert len(c) == 600 and c.groupby(["system_id", "model"]).seed.nunique().eq(5).all()
    fwd = equal_location(c, ["model"], ["rmse", "mae", "r2", "coverage_80"])
    runs = pd.read_csv(ROOT / "results/degradation_paper/tables/T3b_common_domain_fit.csv")
    retro = runs.groupby("model").rmse.mean().rename(index={"khd-full": "kappa_hydra_d"})
    fwd["retrospective_rmse_equal_array"] = fwd.model.map(retro)
    # same weighting as the forward test (four locations equal), so the two columns of Table VI are comparable
    loc = pd.read_csv(TAB / "T0_sites.csv").set_index("system_id").location
    runs = runs.assign(location=runs.system_id.map(loc), model=runs.model.replace({"khd-full": "kappa_hydra_d"}))
    retro_loc = runs.groupby(["model", "location"]).rmse.mean().groupby("model").mean()
    fwd["retrospective_rmse_equal_location"] = fwd.model.map(retro_loc)
    fwd["ratio_equal_location"] = fwd.rmse / fwd.retrospective_rmse_equal_location
    fwd["n_test_hours"] = int(c.drop_duplicates("system_id").n_test.sum())
    fwd.to_csv(TAB / "T22_forward_by_model.csv", index=False)
    print(fwd.round(4).to_string(index=False))
    g = pd.read_csv(IV / "geographic" / "geographic_runs.csv")
    assert len(g) == 600
    held = equal_location(g, ["model", "condition"], ["rmse", "mae", "r2"])
    plr = pd.read_csv(IV / "tables" / "geographic_test_period_plr.csv")
    plr = plr[(plr.model != "rdtools") & (plr.status == "ok")]
    agree = equal_location(plr, ["model", "condition"], ["abs_difference_reference"])
    held = held.merge(agree.rename(columns={"abs_difference_reference": "rate_abs_diff_reference"}),
                      on=["model", "condition"], how="left")
    held.to_csv(TAB / "T23_heldout_location.csv", index=False)
    print(held.round(4).to_string(index=False))
    gs = IV / "gamma_sensitivity.csv"
    if gs.exists():
        s = pd.read_csv(gs)
        s.to_csv(TAB / "T24_field_rate_sensitivity.csv", index=False)
        print(s.head().to_string(index=False))


if __name__ == "__main__":
    main()
