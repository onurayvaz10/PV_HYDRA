"""Reference variants on the 18 measured systems (protocol P4, committed 2026-10-02 before the runs).

  typical    technology-typical temperature coefficient (the reference of the paper)
  listed     coefficient of the installed module in the CEC module database (pvlib SAM CECMod, gamma_r), only
             where the module model given by the data provider (DKA Solar Centre array pages) is listed
  fitted     coefficient fitted to the first 12 months (regression of P/(P_r POA/1000) on T_c - 25, POA 600-1100)
  clearsky   RdTools clear-sky workflow (sensor-independent), where the array geometry is documented

All sensor variants use the fixed hour mask of the unmodified record. Output: results/fixed_mask/reference_variants.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixed_mask as fm  # noqa: E402
from run_semisynthetic_gamma import fitted_gamma  # noqa: E402
from src.degradation.rdtools_pipeline import sensor_plr  # noqa: E402

# Module model per array (DKA Solar Centre array pages, read 2026-10-02) and the CEC database entry, if listed.
DKASC_MODULE = {
    "dkasc_7": ("First Solar FS-272", "First_Solar__Inc__FS_272"),
    "dkasc_8": ("Kaneka G-EA060", None),
    "dkasc_10": ("SunPower SPR-215-WHT-I", "SunPower_SPR_215_WHT_U"),
    "dkasc_11": ("BP 3165N", None),
    "dkasc_12": ("BP 4170N", None),
    "dkasc_13": ("Trina TSM-175DC01", None),
    "dkasc_14": ("Kyocera KD135GX-LP", "Kyocera_Solar_KD135GX_LP"),
    "dkasc_17": ("Sanyo HIP-210NKHE5", None),
    "dkasc_18": ("SunPower SPR-238-WHT-D", "SunPower_SPR_238E_WHT_D"),
    "dkasc_19": ("Sungrid SG-280M6", None),
    "dkasc_20": ("Sungrid SG-280P6", None),
    "dkasc_21": ("Evergreen ES-A-205-fa3", None),
}
# Array geometry: DKASC array pages (tilt 20 deg, solar north) and the PVDAQ catalogue; trackers and roofs without a
# documented tilt are not run with the clear-sky workflow.
GEOMETRY = {**{s: (20.0, 0.0) for s in DKASC_MODULE}, "pvdaq_1423": (35.0, 180.0), "pvdaq_2107": (25.0, 180.0),
            "pvdaq_10": (40.0, 180.0), "pvdaq_33": (40.0, 180.0)}


def clearsky_rate(base: pd.DataFrame, meta: dict, gamma: float, tilt: float, azimuth: float) -> dict:
    import rdtools
    import datetime
    offset = float(meta["std_offset_h"])                     # records are on local standard time
    # Alice Springs (UTC+9:30, no daylight saving) is the IANA zone Australia/Darwin; whole-hour offsets pass as int
    tz = "Australia/Darwin" if offset == 9.5 else int(offset)
    idx = base.index.tz_localize("Australia/Darwin" if offset == 9.5
                                 else datetime.timezone(datetime.timedelta(hours=offset)))
    lat = meta.get("lat", meta.get("latitude"))
    lon = meta.get("lon", meta.get("longitude"))
    data = base.set_axis(idx)
    ta = rdtools.TrendAnalysis(data.power_w.clip(lower=0), poa_global=data.poa_wm2.clip(lower=0),
                               temperature_ambient=data.t_amb_c if "t_amb_c" in data else None,
                               gamma_pdc=gamma, power_dc_rated=meta["dc_kw"] * 1000.0, interp_freq="60min")
    ta.set_clearsky(pvlib_location=pvlib.location.Location(lat, lon, tz=tz), pv_azimuth=azimuth, pv_tilt=tilt)
    # hourly records: the clear-sky-index filter (the pvlib detection needs sub-hourly windows)
    ta.filter_params["clearsky_filter"] = {"model": "csi"}
    ta.clearsky_analysis(analyses=["yoy_degradation"])
    yd = ta.results["clearsky"]["yoy_degradation"]
    daily = ta.clearsky_aggregated_performance[ta.clearsky_filter_aggregated.reindex(
        ta.clearsky_aggregated_performance.index, fill_value=False)].dropna()
    t = (daily.index - daily.index.min()).days.to_numpy() / 365.25   # same YoY centre as sensor_plr
    return {"plr": float(yd["p50_rd"]), "ci_low": float(yd["rd_confidence_interval"][0]),
            "ci_high": float(yd["rd_confidence_interval"][1]),
            "t_center_years": float(np.median(t[t >= 1]) - 0.5) if (t >= 1).any() else np.nan}


def main() -> None:
    cec = pvlib.pvsystem.retrieve_sam("CECMod")
    rows = []
    for sid in fm.systems():
        base, meta, gamma, ym, _ = fm.load(sid)
        out = {"system_id": sid, "module": DKASC_MODULE.get(sid, (meta.get("label", ""), None))[0]}
        r = sensor_plr(base, meta["dc_kw"], gamma, mask=ym)
        out.update(gamma_typical=gamma, plr_typical=r["plr"])
        entry = DKASC_MODULE.get(sid, (None, None))[1]
        if entry:
            g = float(cec[entry]["gamma_r"]) / 100.0
            r = sensor_plr(base, meta["dc_kw"], g, mask=ym)
            out.update(cec_entry=entry, gamma_listed=g, plr_listed=r["plr"])
        g = fitted_gamma(base, meta["dc_kw"])
        r = sensor_plr(base, meta["dc_kw"], g, mask=ym)
        out.update(gamma_fitted=g, plr_fitted=r["plr"])
        if sid in GEOMETRY:
            try:
                cs = clearsky_rate(base, meta, gamma, *GEOMETRY[sid])
                out.update(plr_clearsky=cs["plr"], clearsky_ci_low=cs["ci_low"], clearsky_ci_high=cs["ci_high"])
            except Exception as exc:
                out["clearsky_status"] = f"failed: {type(exc).__name__}: {exc}"[:200]
        else:
            out["clearsky_status"] = "not applicable (tracker or undocumented tilt)"
        rows.append(out)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    fm.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(fm.OUT / "reference_variants.csv", index=False)


def inject_clearsky() -> None:
    """Re-run only the clear-sky part of inject() (first run failed on an attribute name); fills the clear-sky columns."""
    f = fm.OUT / "reference_variants_injection.csv"
    d = pd.read_csv(f)
    for i, r in d.iterrows():
        if r.system_id not in GEOMETRY or pd.notna(r.plr_clearsky):
            continue
        base, meta, gamma, ym, _ = fm.load(r.system_id)
        try:
            cs = clearsky_rate(fm.injected(base, r.injection), meta, gamma, *GEOMETRY[r.system_id])
            d.loc[i, ["plr_clearsky", "t_center_clearsky", "clearsky_status"]] = [cs["plr"], cs["t_center_years"], ""]
        except Exception as exc:
            d.loc[i, "clearsky_status"] = f"failed: {type(exc).__name__}: {exc}"[:200]
        d.to_csv(f, index=False)
        print(r.system_id, r.injection, d.loc[i, "plr_clearsky"], flush=True)


def inject() -> None:
    """Reference variants on the injected records of P1 (same fixed mask): fitted coefficient (refitted on each
    injected record) and the clear-sky workflow. Output: results/fixed_mask/reference_variants_injection.csv"""
    f = fm.OUT / "reference_variants_injection.csv"
    done = set(map(tuple, pd.read_csv(f)[["system_id", "injection"]].astype(str).to_numpy())) if f.exists() else set()
    for sid in fm.systems():
        for inj in fm.INJECTIONS:
            if (sid, str(inj)) in done:
                continue
            base, meta, gamma, ym, _ = fm.load(sid)
            frame = fm.injected(base, inj)
            out = {"system_id": sid, "injection": inj}
            g = fitted_gamma(frame, meta["dc_kw"])
            r = sensor_plr(frame, meta["dc_kw"], g, mask=ym)
            out.update(gamma_fitted=g, plr_fitted=r["plr"], t_center_fitted=r.get("t_center_years"))
            if sid in GEOMETRY:
                try:
                    cs = clearsky_rate(frame, meta, gamma, *GEOMETRY[sid])
                    out.update(plr_clearsky=cs["plr"], t_center_clearsky=cs["t_center_years"])
                except Exception as exc:
                    out["clearsky_status"] = f"failed: {type(exc).__name__}: {exc}"[:200]
            cols = ["system_id", "injection", "gamma_fitted", "plr_fitted", "t_center_fitted", "plr_clearsky",
                    "t_center_clearsky", "clearsky_status"]
            pd.DataFrame([out]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(out, flush=True)


if __name__ == "__main__":
    {"inject": inject, "inject_clearsky": inject_clearsky}.get(sys.argv[1] if sys.argv[1:] else "", main)()
