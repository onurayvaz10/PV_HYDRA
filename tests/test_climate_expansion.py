import unittest

import numpy as np
import pandas as pd
import pvlib

from src.degradation import expansion as ex

META = {"lat": 29.95, "lon": -90.08, "tilt": 30.0, "azimuth": 180.0, "tz": "America/Chicago", "std_offset_h": -6.0,
        "dc_kw": 4.0, "tier": "R", "climate": "Cfa", "system_id": "pvdaq_x", "label": "x"}


def _clearsky_weather(start="2020-01-01", days=30) -> pd.DataFrame:
    """ERA5-like frame: UTC interval-start labels with clear-sky GHI/DNI/DHI at the hour midpoint."""
    labels = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    cs = pvlib.location.Location(META["lat"], META["lon"]).get_clearsky(labels + pd.Timedelta(minutes=30))
    return pd.DataFrame({"shortwave_radiation": cs.ghi.to_numpy(), "direct_normal_irradiance": cs.dni.to_numpy(),
                         "diffuse_radiation": cs.dhi.to_numpy()}, index=labels)


class ExpansionTests(unittest.TestCase):
    def test_transposition_is_physical(self):
        weather = _clearsky_weather()
        poa = ex.transpose(weather, META)
        self.assertTrue((poa >= 0).all())
        night = weather.shortwave_radiation == 0
        self.assertTrue((poa[night] == 0).all())
        # January, equator-facing 30 deg tilt at 30 N: the plane collects more than the horizontal at noon
        noon = weather.shortwave_radiation.idxmax()
        self.assertGreater(poa[noon], weather.shortwave_radiation[noon])

    def test_convention_detects_local_standard(self):
        weather = _clearsky_weather(days=60)
        poa = ex.transpose(weather, META)
        local = (poa.index + pd.Timedelta(hours=META["std_offset_h"])).tz_localize(None)
        signal = pd.Series(4000 * poa.to_numpy() / 1000, index=local)
        res = ex._convention_scores(signal, lambda labels: ex.tier_a.era5_at(poa, labels), META, threshold=80)
        self.assertEqual(res["convention"], "local_standard")

    def test_select_main_counts_existing_systems(self):
        screen = pd.DataFrame({"system_id": ["a", "b", "c", "d"], "tier": "S", "location": "Golden, CO",
                               "climate": "Dfb", "decision": ["pass", "pass", "pass", "fail"],
                               "valid_years": [9.0, 5.0, 7.0, 10.0], "archived_years": [10, 6, 8, 11]})
        sel = ex.select_main(screen, {"Golden, CO": 1}).set_index("system_id")
        self.assertEqual(list(sel.index), ["a", "c", "b"])                  # by record length; failed d dropped
        self.assertEqual(list(sel.role), ["main", "main", "sensitivity"])  # 3 - 1 existing = 2 slots
        self.assertTrue((sel.status == "final").all())

    def test_tier_r_scale_rule_is_final_while_era5_pending(self):
        idx = pd.date_range("2020-01-01", periods=24 * 400, freq="h")
        labels = (idx - pd.Timedelta(hours=META["std_offset_h"])).tz_localize("UTC")
        cs = ex.clearsky_poa(META, labels)
        frame = pd.DataFrame({"power_w": 5.0 * META["dc_kw"] * cs}, index=idx)   # 5 x the rating: unit error
        res = ex.screen_r(frame, {**META, "climate": "pending: raster"}, final_clock=False)
        self.assertEqual(res["decision"], "fail")
        self.assertIn("1.10 x DC", res["reason"])
        ok = ex.screen_r(frame.assign(power_w=0.8 * META["dc_kw"] * cs), META, final_clock=False)
        self.assertEqual(ok["decision"], "pending")                         # never passed on an incomplete screen


class ExpansionSummaryTests(unittest.TestCase):
    def test_tiers_separate_and_locations_weighted_equally(self):
        from scripts.summarize_climate_expansion import by_tier, per_method, recovery
        from src.degradation.injection_truth import expected_for
        rows = []
        # Tier S: location A has two systems (errors 0.1 and 0.3), location B one (error 0.0);
        # Tier R: one location with error 0.2. Baseline rate d = 0 so the exact expected change is r.
        for sid, tier, loc, err in (("a1", "S", "A", 0.1), ("a2", "S", "A", 0.3), ("b1", "S", "B", 0.0),
                                    ("r1", "R", "C", 0.2)):
            base = {"system_id": sid, "tier": tier, "role": "main", "location": loc, "climate": "X",
                    "t_center_years": 3.0}
            rows.append({**base, "injection": 0.0, "plr": 0.0})
            rows.append({**base, "injection": -1.0, "plr": expected_for("rdtools", sid, -1.0, 0.0, 3.0) + err})
        rd = pd.DataFrame(rows)
        rec = recovery(per_method({"rd": rd, "kd": pd.DataFrame(), "ml": pd.DataFrame()}))
        tiers = by_tier(rec).set_index("tier")
        self.assertAlmostEqual(tiers.loc["S", "mae_equal_location"], (0.2 + 0.0) / 2)   # not (0.1+0.3+0)/3
        self.assertAlmostEqual(tiers.loc["R", "mae_equal_location"], 0.2)
        self.assertEqual(int(tiers.loc["S", "n_locations"]), 2)


if __name__ == "__main__":
    unittest.main()
