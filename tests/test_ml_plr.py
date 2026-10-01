import unittest

import numpy as np
import pandas as pd

from src.degradation import ml_plr, perf_models as pm
from src.degradation.rdtools_pipeline import cell_temperature, sensor_plr


def _synthetic(years: float = 5.0, plr: float = -1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=int(years * 8760), freq="h")
    hour = idx.hour.to_numpy() + 0.5
    season = 1 + 0.25 * np.cos(2 * np.pi * (idx.dayofyear.to_numpy() - 172) / 365.25)
    clear = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 1000 * season / 1.25
    cloud = np.repeat(rng.uniform(0.3, 1.0, len(idx) // 24 + 1), 24)[: len(idx)]
    poa = pd.Series(clear * cloud, index=idx)
    tamb = pd.Series(20 + 8 * np.sin(2 * np.pi * (hour - 9) / 24) + 24 * (season - 1), index=idx)
    tcell = cell_temperature(poa, tamb, None)
    t_years = (idx - idx[0]).total_seconds().to_numpy() / (365.25 * 86400)
    power = 5000 * poa / 1000 * (1 - 0.004 * (tcell - 25)) * (1 + plr / 100 * t_years)
    power = power * (1 + rng.normal(0, 0.01, len(idx)))
    return pd.DataFrame({"power_w": power.clip(lower=0), "poa_wm2": poa, "t_amb_c": tamb})


class MlPlrTests(unittest.TestCase):
    def test_reference_end_counts_data_months(self):
        idx = pd.date_range("2020-01-01", "2023-12-31 23:00", freq="h")
        target = pd.Series(1.0, index=idx).where(~((idx >= "2020-03-01") & (idx < "2020-06-01")))
        # three empty months push the end of 24 data months by three calendar months
        self.assertEqual(ml_plr.reference_end(target, months=24), pd.Timestamp("2022-04-01"))

    def test_healthy_hours_drop_outage_days(self):
        frame = _synthetic(years=1.0)
        frame.loc["2015-06-10", "power_w"] = 0.0
        frame.loc["2015-06-11", "power_w"] *= 0.1
        ok = ml_plr.healthy_hours(frame, 5.0)
        self.assertFalse(ok.loc["2015-06-10"].any())
        self.assertFalse(ok.loc["2015-06-11"].any())
        self.assertGreaterEqual(int(ok.loc["2015-06-12"].sum()), 8)

    def test_injection_is_linear_in_time(self):
        frame = _synthetic(years=2.0)
        ratio = (ml_plr.inject(frame, -2.0) / frame.power_w)[frame.power_w > 100]
        self.assertAlmostEqual(ratio.iloc[0], 1.0, places=4)       # first daylight hour: 7 h into the record
        self.assertAlmostEqual(ratio.iloc[-1], 0.96, delta=0.002)

    def test_two_stage_recovers_known_plr(self):
        frame = _synthetic(years=5.0, plr=-1.0)
        d = ml_plr.prepare(frame, 5.0)
        self.assertGreater(int(d["train"].sum()), 5000)
        self.assertGreater(int(d["val"].sum()), 800)
        train = (d["x"][d["train"]], d["y"][d["train"]])
        val = (d["x"][d["val"]], d["y"][d["val"]])
        pred, _ = pm.fit_predict("xgboost", pm.default_params("xgboost"), 0, train, val, d["x"], device="cpu")
        self.assertGreater(ml_plr.scores(pred[d["val"]], d["y"][d["val"]])["r2"], 0.98)
        r = ml_plr.plr_from_prediction(frame, 5.0, pred, d["stamps"])
        self.assertAlmostEqual(r["plr"], -1.0, delta=0.2)
        self.assertAlmostEqual(sensor_plr(frame, 5.0, -0.004)["plr"], -1.0, delta=0.2)

    def test_kappa_hydra_d_recovers_known_plr(self):
        from src.degradation.kappa_hydra_d import estimate
        frame = _synthetic(years=5.0, plr=-1.0)
        r = estimate(frame, 5.0, -0.004, seeds=(0, 1), steps=1500, jackknife=False, device="cpu")
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["plr"], -1.0, delta=0.15)
        self.assertGreater(r["coverage_80"], 0.6)


if __name__ == "__main__":
    unittest.main()
